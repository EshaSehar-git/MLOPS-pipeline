"""
Builds two things the model doesn't produce directly, from fields that
have already been extracted (by either the model or the rule-based path):

  - invoice_number: not one of LayoutLMv3's trained labels (SROIE only has
    company/date/address/total), so it's pulled from the raw OCR text with
    a regex heuristic instead.
  - description: a short, human-readable 1-2 line summary of the receipt,
    built from the fields already on hand — no extra model call needed.
"""
import re
from typing import Dict, Optional

INVOICE_NUMBER_PATTERNS = [
    # "Invoice No: 12345", "TAX INVOICE NO: 19729058", "Receipt # ABC123"
    r"\b(?:tax\s+invoice|invoice|receipt|order|bill)\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Za-z0-9/\-]{3,})",
    # standalone "Inv" abbreviation — \b on both sides so this can't match
    # inside a longer word like "INVOICE" (no word boundary between V/O there)
    r"\binv\b\s*[:;\-]?\s*([A-Za-z0-9/\-]{4,})",
]


def extract_invoice_number(raw_text: str) -> Optional[str]:
    for pattern in INVOICE_NUMBER_PATTERNS:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def build_summary(fields: Dict) -> Dict:
    raw_text = fields.get("raw_text") or ""
    invoice_number = extract_invoice_number(raw_text)

    company = fields.get("company") or "an unknown vendor"
    date = fields.get("date")
    address = fields.get("address")
    total = fields.get("total")

    line1 = f"Receipt from {company}"
    if date:
        line1 += f", dated {date}"
    if total and isinstance(total, dict) and total.get("value"):
        line1 += f", totaling {total['value']}"
    line1 += "."

    line2_parts = []
    line2_parts.append(f"Invoice/receipt #{invoice_number}" if invoice_number else "No invoice number detected")
    if address:
        line2_parts.append(f"issued at {address}")
    line2 = " ".join(line2_parts).strip()
    if not line2.endswith("."):
        line2 += "."

    return {
        "invoice_number": invoice_number,
        "description": f"{line1}\n{line2}",
    }
