import type { ExtractionResponse } from '../types'

export default function ResultCard({ result, title }: { result: ExtractionResponse; title?: string }) {
  const totalText = result.total
    ? `${result.total.value} · ${result.total.confidence.toFixed(1)}% confidence · ${result.total.source}`
    : null

  return (
    <div>
      {title && <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3>}

      <p className="mb-4 whitespace-pre-line rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-sm leading-relaxed text-slate-300">
        {result.description}
      </p>

      <dl className="divide-y divide-slate-800">
        <Field label="Invoice #" value={result.invoice_number} />
        <Field label="Company" value={result.company} />
        <Field label="Date" value={result.date} />
        <Field label="Address" value={result.address} />
        <Field label="Total" value={totalText} />
      </dl>

      <div
        className={`mt-4 rounded-lg border px-3 py-2 text-sm ${
          result.validation_status === 'auto_accept'
            ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
            : 'border-amber-500/30 bg-amber-500/10 text-amber-400'
        }`}
      >
        <strong>{result.validation_status === 'auto_accept' ? 'Auto-accepted' : 'Needs review'}</strong>
        {result.validation_reasons.length > 0 && (
          <ul className="mt-1 list-inside list-disc text-xs opacity-90">
            {result.validation_reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
      </div>

      <details className="mt-4 text-xs">
        <summary className="cursor-pointer text-indigo-400 hover:text-indigo-300">Raw OCR text</summary>
        <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950 p-3 text-slate-400">
          {result.raw_text}
        </pre>
      </details>
    </div>
  )
}

function Field({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2.5">
      <dt className="whitespace-nowrap text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className={`text-right text-sm ${value ? 'text-slate-100' : 'italic text-slate-600'}`}>
        {value || 'not found'}
      </dd>
    </div>
  )
}
