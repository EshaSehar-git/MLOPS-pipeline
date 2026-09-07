import { useRef, useState, type DragEvent } from 'react'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/jpg']

interface Props {
  onFileSelected: (file: File) => void
  previewUrl: string | null
  onExtract: () => void
  onCompare: () => void
  busy: boolean
  hasFile: boolean
  errorMessage: string | null
}

export default function UploadPanel({
  onFileSelected,
  previewUrl,
  onExtract,
  onCompare,
  busy,
  hasFile,
  errorMessage,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0]
    if (!file) return
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setLocalError('Only JPEG/PNG images are supported.')
      return
    }
    setLocalError(null)
    onFileSelected(file)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragOver(false)
    handleFiles(e.dataTransfer.files)
  }

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl shadow-black/20">
      <h2 className="mb-4 text-sm font-semibold text-slate-300">1. Upload a receipt or invoice</h2>

      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-4 transition-colors ${
          dragOver ? 'border-indigo-400 bg-indigo-500/10' : 'border-slate-700 hover:border-slate-600'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
        {previewUrl ? (
          <img src={previewUrl} alt="preview" className="max-h-72 rounded-lg object-contain" />
        ) : (
          <div className="text-center">
            <p className="text-sm text-slate-400">Drag &amp; drop an image here</p>
            <p className="my-2 text-xs text-slate-600">or</p>
            <span className="inline-block rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500">
              Choose file
            </span>
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap gap-3">
        <button
          id="extract-btn"
          onClick={onExtract}
          disabled={!hasFile || busy}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
        >
          {busy ? 'Extracting…' : 'Extract fields'}
        </button>
        <button
          id="compare-btn"
          onClick={onCompare}
          disabled={!hasFile || busy}
          className="rounded-lg border border-indigo-500 px-4 py-2 text-sm font-semibold text-indigo-400 transition hover:bg-indigo-500/10 disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-600"
        >
          Compare model vs. rule-based
        </button>
      </div>

      {(localError || errorMessage) && (
        <p className="mt-3 text-sm text-red-400">{localError || errorMessage}</p>
      )}
    </section>
  )
}
