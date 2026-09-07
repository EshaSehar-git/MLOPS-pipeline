import type { CompareResponse } from '../types'
import ResultCard from './ResultCard'

export default function CompareView({ data }: { data: CompareResponse }) {
  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
      <div>
        {data.model_result ? (
          <ResultCard result={data.model_result} title="LayoutLMv3 model" />
        ) : (
          <ErrorNote title="LayoutLMv3 model" message={data.model_error} />
        )}
      </div>
      <div>
        {data.rule_based_result ? (
          <ResultCard result={data.rule_based_result} title="Rule-based baseline" />
        ) : (
          <ErrorNote title="Rule-based baseline" message={data.rule_based_error} />
        )}
      </div>
    </div>
  )
}

function ErrorNote({ title, message }: { title: string; message: string | null }) {
  return (
    <div>
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3>
      <p className="text-sm text-red-400">{message || 'unknown error'}</p>
    </div>
  )
}
