"""
Converts SROIE's raw OCR + key-info annotations into the token-classification
format LayoutLMv3 expects: each OCR word gets a BIO label (O, B-COMPANY,
B-DATE, B-TOTAL, B-ADDRESS) plus its normalized bounding box.

This version matches the "SROIE2019" Kaggle layout, which looks like:

    SROIE2019/
    ├── train/
    │   ├── box/         # one .txt per receipt: x1,y1,x2,y2,x3,y3,x4,y4,transcript
    │   ├── entities/     # one .txt (JSON) per receipt: {"company":..,"date":..,"address":..,"total":..}
    │   └── img/          # matching .jpg images
    └── test/
        ├── box/
        ├── entities/
        └── img/

(If your download instead uses task1_ocr/ + task2_key_info/ folder names,
use the older version of this script — this one is for the box/entities/img
layout.)

This script:
  1. Reads SROIE's own OCR ground truth (box/*.txt) — no OCR engine needed
     at training-prep time, since SROIE already gives you word boxes.
  2. Matches each OCR word against the known field strings (entities/*.txt)
     to assign a BIO label.
  3. Writes out JSONL files: train/val (from the SROIE "train" split) and
     test (from the SROIE "test" split, used as a fully held-out set).

Usage:
    python src/training/prepare_dataset.py --raw_dir "/content/drive/MyDrive/SROIE2019" --output_dir /content/processed
"""
import argparse
import json
import os
import re
from pathlib import Path
from sklearn.model_selection import train_test_split


LABELS = ["O", "B-COMPANY", "B-DATE", "B-ADDRESS", "B-TOTAL"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def normalize_bbox(box, width, height):
    """LayoutLMv3 expects bounding boxes normalized to a 0-1000 scale."""
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def load_ocr_words(ocr_txt_path: str):
    """
    SROIE task1 ground truth format (per line):
        x1,y1,x2,y2,x3,y3,x4,y4,transcript
    Returns list of (text, [left, top, right, bottom])
    """
    words = []
    with open(ocr_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) < 9:
                continue
            coords = list(map(int, parts[:8]))
            text = parts[8]
            xs = coords[0::2]
            ys = coords[1::2]
            box = [min(xs), min(ys), max(xs), max(ys)]
            words.append((text, box))
    return words


def load_key_info(key_info_path: str):
    """
    entities/*.txt files are usually JSON, but some SROIE mirrors have stray
    trailing commas or encoding quirks. Fall back gracefully instead of
    crashing the whole prep run on one bad file.
    """
    with open(key_info_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # last-resort cleanup: remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}


def assign_labels(words, key_info):
    """
    Naive matching: if a word's text appears inside one of the ground-truth
    field strings, label it accordingly. This is intentionally simple —
    good enough as a first pass; refine with fuzzy matching if you want
    higher label quality.
    """
    labels = []
    field_map = {
        "company": "B-COMPANY",
        "date": "B-DATE",
        "address": "B-ADDRESS",
        "total": "B-TOTAL",
    }
    for text, box in words:
        label = "O"
        for field, tag in field_map.items():
            field_value = str(key_info.get(field, "")).lower()
            if field_value and text.lower() in field_value:
                label = tag
                break
        labels.append(label)
    return labels


def find_image_path(img_dir: Path, stem: str):
    """SROIE images are usually .jpg, but check a couple extensions just in case."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return img_dir / f"{stem}.jpg"  # fallback guess, may not exist


def build_examples_from_split(split_dir: Path):
    """
    split_dir is e.g. .../SROIE2019/train  or  .../SROIE2019/test
    Expects box/, entities/, img/ subfolders inside it.
    """
    box_dir = split_dir / "box"
    entities_dir = split_dir / "entities"
    img_dir = split_dir / "img"

    if not box_dir.exists() or not entities_dir.exists():
        print(f"  WARNING: {box_dir} or {entities_dir} not found — skipping this split")
        return []

    examples = []
    skipped = 0
    for box_file in sorted(box_dir.glob("*.txt")):
        stem = box_file.stem
        entity_file = entities_dir / f"{stem}.txt"
        if not entity_file.exists():
            skipped += 1
            continue

        words_boxes = load_ocr_words(str(box_file))
        key_info = load_key_info(str(entity_file))
        if not words_boxes or not key_info:
            skipped += 1
            continue

        labels = assign_labels(words_boxes, key_info)
        texts = [w[0] for w in words_boxes]
        boxes = [w[1] for w in words_boxes]

        examples.append({
            "id": stem,
            "tokens": texts,
            "bboxes": boxes,        # raw pixel coords; normalized per-image at train time
            "ner_tags": [LABEL2ID[l] for l in labels],
            "image_path": str(find_image_path(img_dir, stem)),
        })

    if skipped:
        print(f"  ({skipped} files skipped — missing entity file, empty OCR, or unparseable JSON)")
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir", required=True,
        help="Path to the SROIE2019 folder (containing train/ and test/ subfolders)"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--val_fraction", type=float, default=0.15,
        help="Fraction of the SROIE 'train' split to hold out as validation"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    print(f"Reading train split from {raw_dir / 'train'}")
    train_pool = build_examples_from_split(raw_dir / "train")
    print(f"  -> {len(train_pool)} examples")

    print(f"Reading test split from {raw_dir / 'test'}")
    test_examples = build_examples_from_split(raw_dir / "test")
    print(f"  -> {len(test_examples)} examples")

    if len(train_pool) == 0:
        raise SystemExit(
            f"No examples found under {raw_dir / 'train'}. "
            f"Check that box/ and entities/ subfolders exist and contain matching .txt files."
        )
    if len(train_pool) < 5:
        raise SystemExit(
            f"Only {len(train_pool)} usable examples found — too few to split into "
            f"train/val. Check the WARNING/skip messages above for parsing issues."
        )

    train_examples, val_examples = train_test_split(
        train_pool, test_size=args.val_fraction, random_state=42
    )

    for split_name, split_data in [
        ("train", train_examples),
        ("val", val_examples),
        ("test", test_examples),
    ]:
        out_path = Path(args.output_dir) / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split_name}: {len(split_data)} examples -> {out_path}")

    with open(Path(args.output_dir) / "label_map.json", "w") as f:
        json.dump(LABEL2ID, f, indent=2)

    print("\nDone. If train/val counts look right but test is 0, that's fine —")
    print("it just means your SROIE2019/test/entities/ folder was empty or missing.")
