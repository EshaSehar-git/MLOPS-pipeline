import type { CompareResponse, ExtractionResponse, ModelInfoResponse } from './types'

async function postFile<T>(path: string, file: File): Promise<T> {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(path, { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch('/model/info')
  if (!res.ok) throw new Error(`Request failed (${res.status})`)
  return res.json()
}

export const extractFields = (file: File) => postFile<ExtractionResponse>('/extract', file)

export const compareExtraction = (file: File) => postFile<CompareResponse>('/extract/compare', file)
