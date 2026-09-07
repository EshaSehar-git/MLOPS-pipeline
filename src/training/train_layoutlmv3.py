"""
Fine-tunes microsoft/layoutlmv3-base for token classification (field
extraction) on the prepared SROIE dataset. Designed to run on a Colab T4 GPU.

Install deps first (not in main requirements.txt since they're heavy and
training-only):
    pip install transformers datasets seqeval accelerate torch pillow mlflow

Usage:
    python src/training/train_layoutlmv3.py \
        --data_dir data/processed \
        --output_dir models/layoutlmv3-sroie \
        --epochs 15 \
        --batch_size 4 \
        --lr 5e-5
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import Dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score, precision_score, recall_score

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def normalize_bbox(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def build_hf_dataset(examples, processor, label_list):
    def gen():
        for ex in examples:
            try:
                image = Image.open(ex["image_path"]).convert("RGB")
            except FileNotFoundError:
                continue
            width, height = image.size
            boxes = [normalize_bbox(b, width, height) for b in ex["bboxes"]]

            encoding = processor(
                image,
                ex["tokens"],
                boxes=boxes,
                word_labels=ex["ner_tags"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            yield {
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "bbox": encoding["bbox"][0],
                "pixel_values": encoding["pixel_values"][0],
                "labels": encoding["labels"][0],
            }

    return Dataset.from_generator(gen)


def compute_metrics(eval_pred, label_list):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def main(args):
    with open(Path(args.data_dir) / "label_map.json") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    label_list = [id2label[i] for i in range(len(id2label))]

    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    train_examples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    val_examples = load_jsonl(Path(args.data_dir) / "val.jsonl")

    train_dataset = build_hf_dataset(train_examples, processor, label_list)
    val_dataset = build_hf_dataset(val_examples, processor, label_list)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to=[],  # we log to mlflow manually below
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: compute_metrics(p, label_list),
    )

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("layoutlmv3-sroie")
        with mlflow.start_run():
            mlflow.log_params({
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "base_model": "microsoft/layoutlmv3-base",
            })
            trainer.train()
            metrics = trainer.evaluate()
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            trainer.save_model(args.output_dir)
            processor.save_pretrained(args.output_dir)
            mlflow.log_artifacts(args.output_dir, artifact_path="model")
    else:
        trainer.train()
        metrics = trainer.evaluate()
        print(metrics)
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    main(args)
