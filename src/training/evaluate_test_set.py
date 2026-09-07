"""
Runs the trained model against the full held-out test set (test.jsonl,
347 examples) and reports precision/recall/F1 — the real accuracy number,
as opposed to eyeballing a few images.

Usage (run from inside the project folder, in Colab or locally with the
model + test data available):
    python src/training/evaluate_test_set.py \
        --model_path /content/drive/MyDrive/layoutlmv3-sroie-model \
        --test_file /content/processed/test.jsonl
"""
import argparse
import json
import torch
from pathlib import Path
from PIL import Image
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor


def normalize_bbox(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def main(args):
    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model_path)
    processor = LayoutLMv3Processor.from_pretrained(args.model_path, apply_ocr=False)
    model.eval()
    id2label = model.config.id2label

    examples = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Evaluating on {len(examples)} test examples...")

    all_true_labels = []
    all_pred_labels = []
    skipped = 0

    for i, ex in enumerate(examples):
        try:
            image = Image.open(ex["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            skipped += 1
            continue

        width, height = image.size
        boxes = [normalize_bbox(b, width, height) for b in ex["bboxes"]]

        encoding = processor(
            image, ex["tokens"], boxes=boxes,
            truncation=True, return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        word_ids = encoding.word_ids(batch_index=0)

        true_tags = [id2label[t] for t in ex["ner_tags"]]

        # align predictions (which are per sub-word token) back to per-word,
        # taking the first sub-token's prediction for each original word
        seen = set()
        pred_tags = ["O"] * len(ex["tokens"])
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen or word_id >= len(pred_tags):
                continue
            seen.add(word_id)
            pred_tags[word_id] = id2label[predictions[idx]]

        all_true_labels.append(true_tags)
        all_pred_labels.append(pred_tags)

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(examples)}")

    if skipped:
        print(f"\n({skipped} examples skipped — image file not found)")

    print("\n=== Overall metrics ===")
    print(f"Precision: {precision_score(all_true_labels, all_pred_labels):.4f}")
    print(f"Recall:    {recall_score(all_true_labels, all_pred_labels):.4f}")
    print(f"F1:        {f1_score(all_true_labels, all_pred_labels):.4f}")

    print("\n=== Per-field breakdown ===")
    print(classification_report(all_true_labels, all_pred_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", required=True)
    args = parser.parse_args()
    main(args)
