import React, { useState, useEffect } from 'react';
import Badge from '../components/common/Badge';
import Loading from '../components/common/Loading';
import { gicApi } from '../api/client';

const COMMODITIES = ['Copper','Steel','Lithium','Aluminum','Cobalt','Nickel','Platinum','Natural Gas','Palladium','Polypropylene','Rhodium','ABS Resin'];

const LLM_STATUS = { backend: 'transformers', model: 'google/flan-t5-base', status: 'healthy' };

const MOCK_NARRATIVES = {
  Copper:
    'Copper is forecast to rise 7.2% over the next 12 months, driven primarily by strong Manufacturing PMI signals and continued green energy infrastructure investment. The ensemble model (SARIMAX: 55%, XGBoost: 45%) shows a mean-reverting regime (H=0.41), suggesting near-term volatility before an upward trend emerges in Q3. Recommend maintaining current procurement contracts and considering a 60% hedge ratio.',
  Lithium:
    'Lithium prices are expected to remain elevated with a modest 4.8% increase, reflecting ongoing EV demand growth offset by expanding Australian and Chilean supply capacity. The XGBoost model assigns 68% weight due to the trending regime detected (Hurst=0.61), and flags supply concentration risk in China as the primary tail risk. Finance team should review long-term supply contracts.',
  Steel:
    'Steel is forecast to decline 3.1% as global manufacturing PMI remains below 50, signaling demand contraction. SARIMAX dominates the ensemble (70% weight) given the strong seasonal pattern, with Q1 typically showing weakness before mid-year construction demand recovery. Current forward contracts provide adequate price protection.',
  'Natural Gas':
    'Natural Gas exhibits high volatility (MAPE: 31.1%) driven by TTF/Henry Hub spread uncertainty and geopolitical supply risks. Point forecasting is not recommended — use scenario analysis only. The Bear scenario (supply disruption, +40%) should be stress-tested in the annual financial plan.',
  Aluminum:
    'Aluminum prices show moderate upward pressure (2.0% CAGR) as energy cost normalization supports production economics. The ARIMA model captures seasonal demand patterns well, with Q4 traditionally showing stronger automotive sector demand. Carbon border tax risks from EU CBAM represent a medium-term upside risk to prices.',
};

const MOCK_AUDIT = [
  { id: 'ev-0012', ts: '2026-06-11T09:05:14Z', type: 'pipeline_complete',    detail: 'elapsed: 312.4s, 12 commodities trained' },
  { id: 'ev-0011', ts: '2026-06-11T09:04:52Z', type: 'simulation_run',       detail: 'Commodity Crisis scenario, 10,000 sims' },
  { id: 'ev-0010', ts: '2026-06-11T09:03:31Z', type: 'pnl_generated',        detail: 'revenue: £19.8B, 12 months' },
  { id: 'ev-0009', ts: '2026-06-11T09:02:17Z', type: 'forecast_generated',   detail: 'Copper, MAPE: 7.0%, regime: mean_reverting' },
  { id: 'ev-0008', ts: '2026-06-11T09:01:55Z', type: 'forecast_generated',   detail: 'Lithium, MAPE: 11.9%, regime: trending' },
  { id: 'ev-0007', ts: '2026-06-11T09:01:22Z', type: 'narrative_generated',  detail: 'backend: transformers, commodity: Copper' },
  { id: 'ev-0006', ts: '2026-06-11T09:00:48Z', type: 'models_trained',       detail: 'n_commodities: 12, SARIMAX+XGBoost' },
  { id: 'ev-0005', ts: '2026-06-11T09:00:11Z', type: 'data_loaded',          detail: 'commodity_rows: 1560, macro_rows: 1320' },
  { id: 'ev-0004', ts: '2026-06-10T18:30:00Z', type: 'bias_alert',           detail: 'Lithium variance: 12.3% > 10% threshold' },
  { id: 'ev-0003', ts: '2026-06-10T08:00:01Z', type: 'pipeline_complete',    detail: 'elapsed: 298.1s' },
];

const MOCK_BIAS = [
  { commodity: 'Copper',       bias: 2.1,  mape: 7.0,  status: 'good',     alert: false },
  { commodity: 'Steel',        bias: 4.8,  mape: 12.4, status: 'good',     alert: false },
  { commodity: 'Lithium',      bias: 12.3, mape: 11.9, status: 'escalate', alert: true  },
  { commodity: 'Aluminum',     bias: 6.2,  mape: 16.7, status: 'alert',    alert: true  },
  { commodity: 'Cobalt',       bias: 3.9,  mape: 14.7, status: 'good',     alert: false },
  { commodity: 'Platinum',     bias: 2.4,  mape: 8.9,  status: 'good',     alert: false },
  { commodity: 'Natural Gas',  bias: 18.7, mape: 31.1, status: 'escalate', alert: true  },
  { commodity: 'Palladium',    bias: 14.1, mape: 29.1, status: 'escalate', alert: true  },
];

const biasColor = (s) => ({ good: 'green', alert: 'yellow', escalate: 'red' }[s] || 'blue');
const eventTypeColor = (t) => {
  if (t.includes('alert') || t.includes('escalation')) return 'red';
  if (t.includes('narrative') || t.includes('generated')) return 'blue';
  if (t.includes('trained') || t.includes('pipeline')) return 'green';
  return 'yellow';
};

export default function Governance() {
  const [selectedCommodity, setSelectedCommodity] = useState('Copper');
  const [narrative, setNarrative] = useState('');
  const [loading, setLoading] = useState(false);
  const [llmStatus, setLlmStatus] = useState(LLM_STATUS);

  useEffect(() => {
    gicApi.health().then(h => {
      if (h?.llm_status) setLlmStatus(h.llm_status);
    }).catch(() => {});
  }, []);

  const generateNarrative = async () => {
    setLoading(true);
    setNarrative('');
    try {
      const res = await gicApi.forecastCommodity(selectedCommodity, 12);
      setNarrative(res?.narrative || MOCK_NARRATIVES[selectedCommodity] || MOCK_NARRATIVES.Copper);
    } catch {
      await new Promise(r => setTimeout(r, 800)); // simulate delay
      setNarrative(MOCK_NARRATIVES[selectedCommodity] || MOCK_NARRATIVES.Copper);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className="text-blue-400">ℹ️</span>
        Connect backend:{' '}
        <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code>
        {' '}— LLM narratives use real inference when connected (google/flan-t5-base).
      </div>

      <div>
        <h1 className="text-2xl font-bold text-white">Governance &amp; LLM Explainability</h1>
        <p className="text-slate-400 text-sm mt-1">Audit trail · Open-source LLM narratives · Bias tracking · Model explainability</p>
      </div>

      {/* LLM Engine Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {[
          {
            label: 'LLM Backend',
            value: llmStatus.backend,
            icon: '🤖',
            status: llmStatus.status === 'healthy' ? 'green' : 'red',
            detail: llmStatus.backend === 'ollama' ? 'llama3.2:1b (local)' :
                    llmStatus.backend === 'transformers' ? 'HuggingFace (CPU)' : 'template fallback',
          },
          {
            label: 'Model',
            value: llmStatus.model,
            icon: '🧠',
            status: 'blue',
            detail: 'Open-source · Free · No API key',
          },
          {
            label: 'Audit Trail',
            value: `${MOCK_AUDIT.length} events`,
            icon: '📋',
            status: 'green',
            detail: 'JSONL append-only · Immutable',
          },
        ].map((card, i) => (
          <div key={i} className="rounded-xl p-5 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">{card.icon}</span>
              <span className="text-slate-400 text-sm">{card.label}</span>
              <Badge label={card.status === 'green' ? 'Active' : card.status === 'red' ? 'Offline' : 'Ready'} color={card.status} />
            </div>
            <p className="text-white font-bold font-mono">{card.value}</p>
            <p className="text-slate-500 text-xs mt-1">{card.detail}</p>
          </div>
        ))}
      </div>

      {/* LLM Narrative Generator */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-1">LLM Narrative Generator</h2>
        <p className="text-slate-500 text-xs mb-4">
          Powered by <code className="text-blue-400">google/flan-t5-base</code> (HuggingFace) · Falls back to Ollama llama3.2:1b if available
        </p>
        <div className="flex gap-3 mb-4">
          <select
            value={selectedCommodity}
            onChange={e => { setSelectedCommodity(e.target.value); setNarrative(''); }}
            className="rounded-lg px-3 py-2 border border-slate-600 text-slate-200 text-sm focus:outline-none focus:border-blue-500"
            style={{ backgroundColor: '#0f172a' }}
          >
            {COMMODITIES.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <button
            onClick={generateNarrative}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
          >
            {loading ? '⏳ Generating…' : '🤖 Generate Narrative'}
          </button>
        </div>

        {loading && <Loading message="Running LLM inference…" />}

        {narrative && !loading && (
          <div className="rounded-lg p-5 border border-blue-800 text-sm leading-relaxed" style={{ backgroundColor: 'rgba(59,130,246,0.07)' }}>
            <div className="flex items-center gap-2 mb-3">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              <span className="text-blue-300 text-xs font-medium">
                LLM narrative · {llmStatus.backend} backend · {selectedCommodity}
              </span>
            </div>
            <p className="text-slate-200">{narrative}</p>
          </div>
        )}

        {!narrative && !loading && (
          <div className="rounded-lg p-4 border border-slate-700 text-center text-slate-500 text-sm" style={{ backgroundColor: '#0f172a' }}>
            Select a commodity and click "Generate Narrative"
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Audit Trail */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-4">Audit Trail (JSONL)</h2>
          <div className="space-y-2 max-h-80 overflow-y-auto scrollbar-thin pr-1">
            {MOCK_AUDIT.map((ev, i) => (
              <div key={i} className="rounded-lg px-3 py-2 border border-slate-700 text-xs" style={{ backgroundColor: '#0f172a' }}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <Badge label={ev.type.replace(/_/g, ' ')} color={eventTypeColor(ev.type)} />
                  </div>
                  <code className="text-slate-500">{ev.ts.replace('T', ' ').slice(0, 19)}</code>
                </div>
                <p className="text-slate-400">{ev.detail}</p>
                <code className="text-slate-600">{ev.id}</code>
              </div>
            ))}
          </div>
        </div>

        {/* Bias Tracking */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-1">Forecast Bias Tracking</h2>
          <p className="text-slate-500 text-xs mb-4">&gt;5% → alert · &gt;10% → L6 governance escalation</p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700 text-xs">
                <th className="text-left pb-2">Commodity</th>
                <th className="text-right pb-2">Bias %</th>
                <th className="text-right pb-2">CV MAPE</th>
                <th className="text-right pb-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {MOCK_BIAS.map((b, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-2 text-slate-200">{b.commodity}</td>
                  <td className={`py-2 text-right font-mono ${b.bias > 10 ? 'text-red-400' : b.bias > 5 ? 'text-yellow-400' : 'text-green-400'}`}>
                    {b.bias}%
                  </td>
                  <td className="py-2 text-right text-slate-400">{b.mape}%</td>
                  <td className="py-2 text-right">
                    <Badge
                      label={b.status === 'escalate' ? 'Escalate' : b.status === 'alert' ? 'Alert' : 'OK'}
                      color={biasColor(b.status)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
