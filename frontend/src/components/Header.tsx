import type { ModelInfoResponse } from '../types'

interface Props {
  modelInfo: ModelInfoResponse | null
  modelInfoError: boolean
}

export default function Header({ modelInfo, modelInfoError }: Props) {
  let tone: 'good' | 'bad' | 'neutral' = 'neutral'
  let label = 'checking model…'

  if (modelInfoError) {
    tone = 'bad'
    label = 'backend unreachable'
  } else if (modelInfo) {
    if (modelInfo.model_available) {
      tone = 'good'
      label = `model: layoutlmv3 (${modelInfo.ocr_engine})`
    } else {
      tone = 'bad'
      label = 'model unavailable — rule-based fallback'
    }
  }

  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-800 bg-slate-900/95 px-6 py-4 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-sm font-bold text-white">
          IE
        </div>
        <div>
          <h1 className="text-base font-semibold text-white">Invoice / Receipt Extraction</h1>
          <p className="text-xs text-slate-400">LayoutLMv3 · MLOps demo</p>
        </div>
      </div>
      <Badge tone={tone}>{label}</Badge>
    </header>
  )
}

function Badge({ tone, children }: { tone: 'good' | 'bad' | 'neutral'; children: React.ReactNode }) {
  const tones: Record<typeof tone, string> = {
    good: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    bad: 'bg-red-500/10 text-red-400 border-red-500/30',
    neutral: 'bg-slate-700/50 text-slate-300 border-slate-600',
  }
  return (
    <span className={`whitespace-nowrap rounded-full border px-3 py-1 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  )
}
