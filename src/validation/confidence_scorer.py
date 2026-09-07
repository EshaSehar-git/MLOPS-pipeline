"""
Takes the extracted fields + OCR confidence scores and decides whether the
result is trustworthy enough to auto-accept, or should be routed to a human
review queue. This is the piece that makes the pipeline "production-grade"
instead of just "a script that sometimes works."
"""
from typing import Dict

# Thresholds — tune these based on validation-set performance
MIN_OCR_CONFIDENCE = 60.0     # below this, OCR itself was uncertain
REQUIRED_FIELDS = ["company", "date", "total"]


def score_extraction(fields: Dict) -> Dict:
    """
    Returns a dict with:
      - status: "auto_accept" | "needs_review"
      - reasons: list of strings explaining any flags
    """
    reasons = []

    # 1. Missing required fields
    for field in REQUIRED_FIELDS:
        if not fields.get(field):
            reasons.append(f"missing_field:{field}")

    # 2. Low OCR confidence on the total (most business-critical field)
    total = fields.get("total")
    if total and isinstance(total, dict):
        if total.get("confidence", 0) < MIN_OCR_CONFIDENCE:
            reasons.append(f"low_confidence_total:{total.get('confidence'):.1f}")
        if total.get("source") == "fallback_max_amount":
            reasons.append("total_used_fallback_heuristic")

    # 3. Sanity check: total should look like a real currency value
    if total and isinstance(total, dict):
        try:
            value = float(str(total.get("value", "")).replace(",", ""))
            if value <= 0 or value > 1_000_000:
                reasons.append(f"suspicious_total_value:{value}")
        except (ValueError, TypeError):
            reasons.append("total_not_parseable")

    status = "needs_review" if reasons else "auto_accept"
    return {"status": status, "reasons": reasons}


if __name__ == "__main__":
    # quick manual test
    example = {
        "company": "Acme Store",
        "date": "12/03/2026",
        "total": {"value": "45.90", "confidence": 42.0, "source": "keyword_match"},
    }
    print(score_extraction(example))
