import { useEffect, useState } from 'react'
import Header from './components/Header'
import UploadPanel from './components/UploadPanel'
import ResultCard from './components/ResultCard'
import CompareView from './components/CompareView'
import { compareExtraction, extractFields, fetchModelInfo } from './api'
import type { CompareResponse, ExtractionResponse, ModelInfoResponse } from './types'

type ResultState =
  | { kind: 'empty' }
  | { kind: 'single'; data: ExtractionResponse }
  | { kind: 'compare'; data: CompareResponse }

export default function App() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [modelInfoError, setModelInfoError] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ResultState>({ kind: 'empty' })

  useEffect(() => {
    fetchModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfoError(true))
  }, [])

  const handleFileSelected = (f: File) => {
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setError(null)
    setResult({ kind: 'empty' })
  }

  const run = async (mode: 'extract' | 'compare') => {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      if (mode === 'extract') {
        setResult({ kind: 'single', data: await extractFields(file) })
      } else {
        setResult({ kind: 'compare', data: await compareExtraction(file) })
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950">
      <Header modelInfo={modelInfo} modelInfoError={modelInfoError} />

      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-6 p-6 lg:grid-cols-[380px_1fr]">
        <UploadPanel
          onFileSelected={handleFileSelected}
          previewUrl={previewUrl}
          onExtract={() => run('extract')}
          onCompare={() => run('compare')}
          busy={busy}
          hasFile={!!file}
          errorMessage={error}
        />

        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl shadow-black/20">
          <h2 className="mb-4 text-sm font-semibold text-slate-300">2. Extracted fields</h2>

          {busy && (
            <div className="flex items-center gap-2 text-sm text-slate-500">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-700 border-t-indigo-500" />
              Extracting…
            </div>
          )}

          {!busy && result.kind === 'empty' && (
            <p className="text-sm text-slate-500">
              Upload an image and click "Extract fields" to see results here.
            </p>
          )}

          {!busy && result.kind === 'single' && <ResultCard result={result.data} />}
          {!busy && result.kind === 'compare' && <CompareView data={result.data} />}
        </section>
      </main>
    </div>
  )
}
