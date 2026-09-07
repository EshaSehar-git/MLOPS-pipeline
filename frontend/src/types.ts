// Mirrors api/schemas.py — keep these in sync with the backend Pydantic models.

export interface TotalField {
  value: string
  confidence: number
  source: string
}

export interface ExtractionResponse {
  invoice_number: string | null
  company: string | null
  date: string | null
  address: string | null
  total: TotalField | null
  description: string
  raw_text: string
  validation_status: 'auto_accept' | 'needs_review'
  validation_reasons: string[]
}

export interface ModelInfoResponse {
  model_available: boolean
  model_path: string
  load_error: string | null
  ocr_engine: string
  extractor: 'layoutlmv3_model' | 'rule_based_fallback'
  fields_extracted: string[]
}

export interface CompareResponse {
  model_result: ExtractionResponse | null
  model_error: string | null
  rule_based_result: ExtractionResponse | null
  rule_based_error: string | null
}
