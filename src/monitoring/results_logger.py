"""
Appends every extraction result to a CSV log — the raw audit trail of
what the API has actually predicted. This is the same kind of data
drift_check.py expects to compare over time (ocr word counts, confidence,
extraction outcomes), so this is where that data comes from.

Logging failures never break the API response — if the CSV write fails
for some reason, it's printed as a warning and the request still succeeds.
"""
import csv
import os
from datetime import datetime, timezone
from typing import Dict

LOG_PATH = os.environ.get("RESULTS_LOG_PATH", os.path.join("logs", "extraction_log.csv"))

FIELDNAMES = [
    "timestamp",
    "filename",
    "invoice_number",
    "company",
    "date",
    "address",
    "total_value",
    "total_confidence",
    "description",
    "extractor",
    "ocr_engine",
    "validation_status",
    "validation_reasons",
    "num_words",
]


def log_extraction(filename: str, extractor: str, ocr_engine: str, response: Dict) -> None:
    """`response` is the dict already built by api/main.py's _to_response()
    — invoice_number, company, date, address, total, description, raw_text,
    validation_status, validation_reasons."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        is_new = not os.path.exists(LOG_PATH)

        total = response.get("total") or {}
        raw_text = response.get("raw_text") or ""

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "invoice_number": response.get("invoice_number") or "",
            "company": response.get("company") or "",
            "date": response.get("date") or "",
            "address": response.get("address") or "",
            "total_value": total.get("value", ""),
            "total_confidence": total.get("confidence", ""),
            "description": (response.get("description") or "").replace("\n", " "),
            "extractor": extractor,
            "ocr_engine": ocr_engine,
            "validation_status": response.get("validation_status", ""),
            "validation_reasons": "|".join(response.get("validation_reasons", [])),
            "num_words": len(raw_text.split()),
        }

        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        print(f"[warning] failed to write extraction log: {e}")
