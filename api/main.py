"""
FastAPI service for the invoice/receipt extraction pipeline.

Run locally:
    uvicorn api.main:app --reload --port 8000

Then:
    curl -X POST "http://localhost:8000/extract" -F "file=@data/sample_receipts/receipt1.jpg"

By default this tries to use the fine-tuned LayoutLMv3 model at
models/layoutlmv3-sroie/ (or the path set in the MODEL_PATH env var). If the
model isn't found there, it automatically falls back to the zero-training
rule-based extractor so the API still works.

Endpoints:
    GET  /              the frontend (upload a receipt, see extracted fields)
    GET  /health          liveness + whether the model is loaded
    GET  /model/info      which extractor/OCR engine is active, and why
    POST /extract         extract fields from a single receipt/invoice image
    POST /extract/batch   same, for multiple images in one request
    POST /extract/compare run both the model and the rule-based baseline on
                           the same image, side by side (useful for judging
                           whether the fine-tuned model is worth deploying)
"""
import os
import shutil
import sys
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.extraction.field_parser import parse_fields as parse_fields_rule_based
from src.inference.model_inference import (
    extract_with_model,
    model_is_available,
    model_load_error,
    MODEL_PATH,
    OCR_ENGINE,
)
from src.validation.confidence_scorer import score_extraction
from src.extraction.summary import build_summary
from src.monitoring.results_logger import log_extraction
from api.schemas import (
    ExtractionResponse,
    HealthResponse,
    ModelInfoResponse,
    BatchExtractionResponse,
    CompareResponse,
)

app = FastAPI(
    title="Invoice/Receipt Extraction API",
    description="OCR + field extraction pipeline for receipts and invoices",
    version="0.1.0",
)

SUPPORTED_CONTENT_TYPES = ("image/jpeg", "image/png", "image/jpg")


def _save_upload(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def _extract_fields(tmp_path: str) -> tuple[dict, str]:
    """Returns (fields, extractor_name) — the extractor name is what actually
    ran, which may differ from OCR_ENGINE/MODEL_PATH if the model silently
    fell back, and is recorded in the extraction log for that reason."""
    if model_is_available():
        return extract_with_model(tmp_path), "layoutlmv3_model"
    print(f"[warning] {model_load_error()} — falling back to rule-based extractor")
    return parse_fields_rule_based(tmp_path, engine=OCR_ENGINE), "rule_based_fallback"


def _to_response(fields: dict) -> dict:
    validation = score_extraction(fields)
    summary = build_summary(fields)
    return {
        "invoice_number": summary["invoice_number"],
        "company": fields.get("company"),
        "date": fields.get("date"),
        "address": fields.get("address"),
        "total": fields.get("total"),
        "description": summary["description"],
        "raw_text": fields.get("raw_text", ""),
        "validation_status": validation["status"],
        "validation_reasons": validation["reasons"],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": model_is_available()}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    available = model_is_available()
    return {
        "model_available": available,
        "model_path": MODEL_PATH,
        "load_error": None if available else model_load_error(),
        "ocr_engine": OCR_ENGINE,
        "extractor": "layoutlmv3_model" if available else "rule_based_fallback",
        "fields_extracted": ["company", "date", "address", "total"],
    }


@app.post("/extract", response_model=ExtractionResponse)
async def extract(file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")

    tmp_path = _save_upload(file)
    try:
        fields, extractor = _extract_fields(tmp_path)
        response = _to_response(fields)
        log_extraction(file.filename, extractor, OCR_ENGINE, response)
        return response
    finally:
        os.remove(tmp_path)


@app.post("/extract/batch", response_model=BatchExtractionResponse)
async def extract_batch(files: List[UploadFile] = File(...)):
    """
    Extracts fields from multiple images in one request. Each file is
    processed independently — one bad/corrupt file doesn't fail the batch,
    it just shows up with an "error" set on its own result.
    """
    results = []
    succeeded = 0
    for file in files:
        if file.content_type not in SUPPORTED_CONTENT_TYPES:
            results.append({"filename": file.filename, "error": "unsupported_content_type"})
            continue

        tmp_path = _save_upload(file)
        try:
            fields, extractor = _extract_fields(tmp_path)
            response = _to_response(fields)
            log_extraction(file.filename, extractor, OCR_ENGINE, response)
            results.append({"filename": file.filename, "error": None, **response})
            succeeded += 1
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
        finally:
            os.remove(tmp_path)

    return {
        "results": results,
        "total": len(files),
        "succeeded": succeeded,
        "failed": len(files) - succeeded,
    }


@app.post("/extract/compare", response_model=CompareResponse)
async def extract_compare(file: UploadFile = File(...)):
    """
    Runs both the LayoutLMv3 model and the rule-based baseline on the same
    image so you can compare them directly — useful while judging whether
    the fine-tuned model is actually beating the zero-training baseline.
    """
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")

    tmp_path = _save_upload(file)
    response = {}
    try:
        if model_is_available():
            try:
                model_fields = extract_with_model(tmp_path)
                model_response = _to_response(model_fields)
                log_extraction(file.filename, "layoutlmv3_model", OCR_ENGINE, model_response)
                response["model_result"] = model_response
            except Exception as e:
                response["model_error"] = str(e)
        else:
            response["model_error"] = model_load_error()

        try:
            rule_fields = parse_fields_rule_based(tmp_path, engine=OCR_ENGINE)
            rule_response = _to_response(rule_fields)
            log_extraction(file.filename, "rule_based_fallback", OCR_ENGINE, rule_response)
            response["rule_based_result"] = rule_response
        except Exception as e:
            response["rule_based_error"] = str(e)
    finally:
        os.remove(tmp_path)

    return response


# Serves the built React app (frontend/dist, produced by `npm run build`).
# Mounted last and at "/" so it only catches requests that don't match one
# of the API routes above — Starlette checks routes in registration order.
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
