from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any


class TotalField(BaseModel):
    value: str
    confidence: float
    source: str


class ExtractionResponse(BaseModel):
    invoice_number: Optional[str] = None
    company: Optional[str] = None
    date: Optional[str] = None
    address: Optional[str] = None
    total: Optional[TotalField] = None
    description: str                # short 1-2 line human-readable summary
    raw_text: str
    validation_status: str          # "auto_accept" | "needs_review"
    validation_reasons: List[str] = []


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_available: bool
    model_path: str
    load_error: Optional[str] = None
    ocr_engine: str
    extractor: str                  # "layoutlmv3_model" | "rule_based_fallback"
    fields_extracted: List[str]


class BatchExtractionItem(BaseModel):
    filename: str
    error: Optional[str] = None
    invoice_number: Optional[str] = None
    company: Optional[str] = None
    date: Optional[str] = None
    address: Optional[str] = None
    total: Optional[TotalField] = None
    description: Optional[str] = None
    raw_text: Optional[str] = None
    validation_status: Optional[str] = None
    validation_reasons: List[str] = []


class BatchExtractionResponse(BaseModel):
    results: List[BatchExtractionItem]
    total: int
    succeeded: int
    failed: int


class CompareResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_result: Optional[ExtractionResponse] = None
    model_error: Optional[str] = None
    rule_based_result: Optional[ExtractionResponse] = None
    rule_based_error: Optional[str] = None
