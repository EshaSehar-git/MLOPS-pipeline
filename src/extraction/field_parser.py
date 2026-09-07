"""
Rule-based / regex field extractor. This is the zero-training baseline —
it takes raw OCR text and pulls out company, date, address, and total using
patterns. Good enough to get the full pipeline (API, Docker, monitoring)
working before you invest in fine-tuning LayoutLMv3.

Usage:
    python src/extraction/field_parser.py --image data/sample_receipts/receipt1.jpg
"""
import argparse
import re
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ocr.extract_text import extract_text, words_to_plain_text, OCRWord
from typing import List, Dict, Optional

DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    r"\b\d{1,2}\s?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s?\d{2,4}\b",
]

TOTAL_KEYWORDS = ["total", "amount due", "grand total", "balance due"]

CURRENCY_AMOUNT = r"(?:RM|\$|USD|MYR)?\s?\d{1,3}(?:,\d{3})*\.\d{2}"

POSTAL_CODE = r"\b\d{4,6}\b"


def extract_date(text: str) -> Optional[str]:
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def extract_total(words: List[OCRWord]) -> Optional[Dict]:
    """
    Looks for a TOTAL keyword, then searches nearby words (same row, to the
    right) for a currency-formatted amount. Falls back to the largest
    currency amount found anywhere in the document.
    """
    text_lower = [w.text.lower() for w in words]

    for i, word_text in enumerate(text_lower):
        if any(keyword in word_text for keyword in TOTAL_KEYWORDS):
            anchor = words[i]
            # search words on roughly the same horizontal line, to the right
            same_row_candidates = [
                w for w in words
                if abs(w.top - anchor.top) < anchor.height * 1.5 and w.left >= anchor.left
            ]
            for w in same_row_candidates:
                match = re.search(CURRENCY_AMOUNT, w.text)
                if match:
                    return {"value": match.group(0), "confidence": w.confidence, "source": "keyword_match"}

    # fallback: largest currency-looking number in the doc
    amounts = []
    for w in words:
        match = re.search(CURRENCY_AMOUNT, w.text)
        if match:
            cleaned = re.sub(r"[^\d.]", "", match.group(0))
            try:
                amounts.append((float(cleaned), w))
            except ValueError:
                continue
    if amounts:
        amounts.sort(key=lambda x: x[0], reverse=True)
        value, w = amounts[0]
        return {"value": f"{value:.2f}", "confidence": w.confidence, "source": "fallback_max_amount"}

    return None


def extract_company(words: List[OCRWord]) -> Optional[str]:
    """
    Heuristic: company name is usually one of the first few lines, in the
    top portion of the receipt, often the largest/boldest text. Here we
    just take the first non-empty line as a simple baseline.
    """
    if not words:
        return None
    sorted_words = sorted(words, key=lambda w: (w.top, w.left))
    top_region = [w for w in sorted_words if w.top < sorted_words[0].top + 100]
    if top_region:
        return " ".join(w.text for w in top_region[:5])
    return None


def extract_address(words: List[OCRWord]) -> Optional[str]:
    """
    Heuristic: receipt addresses usually span 1-2 lines just below the
    company name and contain a postal/zip code. Group words into rows by
    vertical position, then return the first row with a postal-code-like
    number plus the row before it (street part often wraps).
    """
    if not words:
        return None

    sorted_words = sorted(words, key=lambda w: (w.top, w.left))
    rows: List[List[OCRWord]] = []
    for w in sorted_words:
        if rows and abs(w.top - rows[-1][0].top) < w.height:
            rows[-1].append(w)
        else:
            rows.append([w])

    for i, row in enumerate(rows):
        row_text = " ".join(w.text for w in row)
        if re.search(POSTAL_CODE, row_text):
            if i > 0:
                prev_text = " ".join(w.text for w in rows[i - 1])
                return f"{prev_text} {row_text}".strip()
            return row_text
    return None


def parse_fields(image_path: str, engine: str = "tesseract") -> Dict:
    words = extract_text(image_path, engine=engine)
    full_text = words_to_plain_text(words)

    result = {
        "company": extract_company(words),
        "date": extract_date(full_text),
        "address": extract_address(words),
        "total": extract_total(words),
        "raw_text": full_text,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--engine", default="tesseract", choices=["tesseract", "easyocr"])
    args = parser.parse_args()

    fields = parse_fields(args.image, engine=args.engine)
    print(json.dumps(fields, indent=2))
