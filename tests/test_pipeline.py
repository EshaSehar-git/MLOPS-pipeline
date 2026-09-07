import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.validation.confidence_scorer import score_extraction
from src.extraction.field_parser import extract_date, extract_total
from src.ocr.extract_text import OCRWord


def test_extract_date_slash_format():
    text = "Purchase date: 12/03/2026 thank you"
    assert extract_date(text) == "12/03/2026"


def test_extract_date_none_found():
    text = "no dates here at all"
    assert extract_date(text) is None


def test_score_extraction_auto_accept():
    fields = {
        "company": "Acme Store",
        "date": "12/03/2026",
        "total": {"value": "45.90", "confidence": 92.0, "source": "keyword_match"},
    }
    result = score_extraction(fields)
    assert result["status"] == "auto_accept"
    assert result["reasons"] == []


def test_score_extraction_needs_review_missing_field():
    fields = {"company": None, "date": "12/03/2026", "total": None}
    result = score_extraction(fields)
    assert result["status"] == "needs_review"
    assert "missing_field:company" in result["reasons"]
    assert "missing_field:total" in result["reasons"]


def test_score_extraction_low_confidence_flagged():
    fields = {
        "company": "Acme Store",
        "date": "12/03/2026",
        "total": {"value": "45.90", "confidence": 20.0, "source": "keyword_match"},
    }
    result = score_extraction(fields)
    assert result["status"] == "needs_review"
    assert any("low_confidence_total" in r for r in result["reasons"])


def test_extract_total_from_words():
    words = [
        OCRWord(text="Total:", left=10, top=100, width=40, height=15, confidence=90),
        OCRWord(text="$45.90", left=60, top=100, width=40, height=15, confidence=88),
    ]
    result = extract_total(words)
    assert result is not None
    assert "45.90" in result["value"]
