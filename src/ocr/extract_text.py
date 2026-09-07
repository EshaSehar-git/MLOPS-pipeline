"""
OCR wrapper. Supports Tesseract (fast, lightweight) and EasyOCR (better on
noisy/rotated scans, but heavier). Both run on CPU fine for this project size.

Usage:
    python src/ocr/extract_text.py --image data/sample_receipts/receipt1.jpg
    python src/ocr/extract_text.py --image data/sample_receipts/receipt1.jpg --engine easyocr
"""
import argparse
from dataclasses import dataclass, asdict
from typing import List
import json

from PIL import Image


@dataclass
class OCRWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float  # 0-100


def run_tesseract(image_path: str) -> List[OCRWord]:
    import pytesseract
    from pytesseract import Output

    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        words.append(
            OCRWord(
                text=text,
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
                confidence=float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0,
            )
        )
    return words


def run_easyocr(image_path: str) -> List[OCRWord]:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(image_path)  # [(bbox, text, confidence), ...]

    words = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        left, top = int(min(xs)), int(min(ys))
        width, height = int(max(xs) - left), int(max(ys) - top)
        words.append(
            OCRWord(
                text=text.strip(),
                left=left,
                top=top,
                width=width,
                height=height,
                confidence=float(conf) * 100,
            )
        )
    return words


def sort_reading_order(words: List[OCRWord]) -> List[OCRWord]:
    """
    Groups words into visual rows and sorts each row left-to-right, rows
    top-to-bottom. Tesseract already returns words in roughly this order;
    EasyOCR does not — it returns detection order, which reads as scrambled
    nonsense once joined into plain text.

    Rows are built by comparing each word only to the *previous* word in
    top-sorted order (a new row starts when the gap exceeds a fraction of
    line height) — not by merging into an expanding row bounding box, which
    on dense receipts "chains" adjacent lines together into one blob once
    two rows happen to overlap by even a couple of pixels.
    """
    if not words:
        return words

    sorted_words = sorted(words, key=lambda w: w.top)
    rows: List[List[OCRWord]] = [[sorted_words[0]]]
    for prev, w in zip(sorted_words, sorted_words[1:]):
        threshold = min(prev.height, w.height) * 0.6
        if w.top - prev.top > threshold:
            rows.append([w])
        else:
            rows[-1].append(w)

    ordered: List[OCRWord] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda w: w.left))
    return ordered


def extract_text(image_path: str, engine: str = "tesseract") -> List[OCRWord]:
    if engine == "tesseract":
        words = run_tesseract(image_path)
    elif engine == "easyocr":
        words = run_easyocr(image_path)
    else:
        raise ValueError(f"Unknown OCR engine: {engine}")
    return sort_reading_order(words)


def words_to_plain_text(words: List[OCRWord]) -> str:
    return " ".join(w.text for w in words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to receipt/invoice image")
    parser.add_argument("--engine", default="tesseract", choices=["tesseract", "easyocr"])
    args = parser.parse_args()

    ocr_words = extract_text(args.image, engine=args.engine)
    print(json.dumps([asdict(w) for w in ocr_words], indent=2))
    print("\n--- Plain text ---")
    print(words_to_plain_text(ocr_words))
