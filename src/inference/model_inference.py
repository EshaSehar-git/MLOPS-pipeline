"""
Loads the fine-tuned LayoutLMv3 model once (module-level singleton) and
exposes a single function, extract_with_model(), that mirrors the same
output shape as the rule-based src/extraction/field_parser.py — so the API
and validation layer don't need to change regardless of which one is used.

Set MODEL_PATH via the MODEL_PATH environment variable, or edit the default
below to match wherever your downloaded model folder actually lives.
"""
import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ocr.extract_text import extract_text

# Change this default if your model folder has a different name/location,
# or set the MODEL_PATH environment variable instead of editing this file.
DEFAULT_MODEL_PATH = "models/layoutlmv3-sroie"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

# "tesseract" (default, used in the Docker image) or "easyocr" (no system
# binary required — set OCR_ENGINE=easyocr for local dev without Tesseract).
OCR_ENGINE = os.environ.get("OCR_ENGINE", "tesseract")

# Below this, a single token labeled as a field is more likely a stray
# misclassification than a real hit — tune based on validation-set performance.
MIN_FIELD_CONFIDENCE = 0.5
# How many un-labeled word positions can be skipped and still count as the
# same contiguous span (e.g. a stopword the model didn't tag in the middle
# of an address). Bigger gaps are treated as a separate occurrence of the
# field elsewhere in the document.
MAX_SPAN_GAP = 2

_model = None
_processor = None
_load_error = None


def _lazy_load():
    """Loads the model on first use, not at import time (keeps startup fast
    and lets the API still boot even if the model isn't in place yet)."""
    global _model, _processor, _load_error
    if _model is not None or _load_error is not None:
        return

    if not os.path.isdir(MODEL_PATH):
        _load_error = (
            f"MODEL_PATH '{MODEL_PATH}' does not exist. "
            f"Set the MODEL_PATH environment variable or move your model there."
        )
        return

    try:
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
        _model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
        _processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)
        _model.eval()
    except Exception as e:
        _load_error = f"Failed to load model from '{MODEL_PATH}': {e}"


def model_is_available() -> bool:
    _lazy_load()
    return _model is not None


def model_load_error() -> str:
    _lazy_load()
    return _load_error or ""


def extract_with_model(image_path: str) -> dict:
    """
    Returns the same shape as src/extraction/field_parser.py's parse_fields():
        {
          "company": str | None,
          "date": str | None,
          "address": str | None,
          "total": {"value": str, "confidence": float, "source": str} | None,
          "raw_text": str,
        }
    """
    import torch

    _lazy_load()
    if _model is None:
        raise RuntimeError(model_load_error())

    # 1. OCR the image ourselves so we know exact words + boxes
    ocr_words = extract_text(image_path, engine=OCR_ENGINE)
    if not ocr_words:
        return {"company": None, "date": None, "total": None, "raw_text": ""}

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    words = [w.text for w in ocr_words]
    boxes = [
        [
            int(1000 * w.left / width),
            int(1000 * w.top / height),
            int(1000 * (w.left + w.width) / width),
            int(1000 * (w.top + w.height) / height),
        ]
        for w in ocr_words
    ]

    # 2. Run the model
    encoding = _processor(image, words, boxes=boxes, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = _model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    predictions = probs.argmax(-1).tolist()
    confidences = probs.max(-1).values.tolist()
    word_ids = encoding.word_ids(batch_index=0)
    id2label = _model.config.id2label

    # 3. Merge sub-word predictions back into whole words
    seen = set()
    word_predictions = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        label = id2label[predictions[idx]]
        if label != "O":
            word_predictions[word_id] = (label, confidences[idx])

    # 4. Group by field
    fields = defaultdict(list)
    for word_id, (label, conf) in word_predictions.items():
        field_name = label.replace("B-", "").replace("I-", "")
        fields[field_name].append((word_id, words[word_id], conf))

    def best_span(name):
        """
        Same-labeled tokens aren't necessarily one continuous phrase — e.g.
        a company name often appears twice on a receipt (header + footer),
        and naively joining every occurrence in document order produces
        nonsense like "ACME CORP THANK YOU FOR SHOPPING AT ACME CORP".
        So: drop low-confidence single-token noise, split what's left into
        contiguous runs (allowing a small gap for unlabeled connector
        words), and keep only the highest-confidence run.
        """
        items = [item for item in fields.get(name, []) if item[2] >= MIN_FIELD_CONFIDENCE]
        if not items:
            return None
        items_sorted = sorted(items, key=lambda x: x[0])

        spans = [[items_sorted[0]]]
        for prev, curr in zip(items_sorted, items_sorted[1:]):
            if curr[0] - prev[0] <= MAX_SPAN_GAP:
                spans[-1].append(curr)
            else:
                spans.append([curr])

        return max(spans, key=lambda span: sum(c for _, _, c in span) / len(span))

    def join_field(name):
        span = best_span(name)
        if span is None:
            return None
        return " ".join(text for _, text, _ in span)

    def total_field():
        span = best_span("TOTAL")
        if span is None:
            return None
        # within the chosen span, the single highest-confidence token is the
        # actual amount (multi-word spans here would just be OCR noise)
        best = max(span, key=lambda x: x[2])
        return {
            "value": best[1],
            "confidence": round(best[2] * 100, 2),  # scale to 0-100 like the OCR confidence used elsewhere
            "source": "layoutlmv3_model",
        }

    return {
        "company": join_field("COMPANY"),
        "date": join_field("DATE"),
        "address": join_field("ADDRESS"),
        "total": total_field(),
        "raw_text": " ".join(words),
    }
