"""
Generates a data drift report comparing a reference dataset (e.g., your
training data / launch-week predictions) against a current batch (e.g., this
week's incoming receipts). Useful for catching things like: "a new store
template broke OCR confidence" or "extraction quality silently degraded."

Usage:
    python src/monitoring/drift_check.py \
        --reference data/processed/train_summary.csv \
        --current data/processed/new_batch_summary.csv \
        --output reports/drift_report.html

Expected CSV columns (one row per processed receipt):
    ocr_confidence_avg, num_words, total_value, extraction_status
"""
import argparse
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def run_drift_report(reference_path: str, current_path: str, output_path: str):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_path)
    print(f"Drift report saved to {output_path}")

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    print(f"Dataset drift detected: {drift_detected}")
    return drift_detected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default="reports/drift_report.html")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_drift_report(args.reference, args.current, args.output)
