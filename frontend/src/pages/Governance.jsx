import React, { useMemo, useState, useEffect } from 'react';
import Badge from '../components/common/Badge';
import Loading from '../components/common/Loading';
import LockedButton from '../components/common/LockedButton';
import { gicApi } from '../api/client';
import { useAuth } from '../auth/AuthContext';
import { can, PERMISSIONS } from '../auth/permissions';

const COMMODITIES = ['Copper', 'Steel', 'Lithium', 'Aluminum', 'Cobalt', 'Nickel', 'Platinum', 'Natural Gas', 'Palladium', 'Polypropylene', 'Rhodium', 'ABS Resin'];

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
  { id: 'ev-0012', ts: '2026-06-11T09:05:14Z', type: 'pipeline_complete',    detail: 'elapsed: 312.4s, 12 commodities trained', actor: 'admin@gic',  hash: '9f2a…c41b', sensitive: true },
  { id: 'ev-0011', ts: '2026-06-11T09:04:52Z', type: 'simulation_run',       detail: 'Commodity Crisis scenario, 10,000 sims',  actor: 'admin@gic',  hash: '3b81…77de', sensitive: true },
  { id: 'ev-0010', ts: '2026-06-11T09:03:31Z', type: 'pnl_generated',        detail: 'revenue: £19.8B, 12 months',             actor: 'system',     hash: 'aa10…0e2f', sensitive: false },
  { id: 'ev-0009', ts: '2026-06-11T09:02:17Z', type: 'forecast_generated',   detail: 'Copper, MAPE: 7.0%, regime: mean_reverting', actor: 'system',  hash: 'd4c9…b801', sensitive: false },
  { id: 'ev-0008', ts: '2026-06-11T09:01:55Z', type: 'forecast_generated',   detail: 'Lithium, MAPE: 11.9%, regime: trending', actor: 'system',     hash: '71ee…39ac', sensitive: false },
  { id: 'ev-0007', ts: '2026-06-11T09:01:22Z', type: 'narrative_generated',  detail: 'backend: transformers, commodity: Copper', actor: 'admin@gic', hash: '5fa2…11cd', sensitive: true },
  { id: 'ev-0006', ts: '2026-06-11T09:00:48Z', type: 'models_trained',       detail: 'n_commodities: 12, SARIMAX+XGBoost',     actor: 'system',     hash: '0c33…ee90', sensitive: false },
  { id: 'ev-0005', ts: '2026-06-11T09:00:11Z', type: 'data_loaded',          detail: 'commodity_rows: 1560, macro_rows: 1320', actor: 'system',     hash: 'b6d7…42aa', sensitive: false },
  { id: 'ev-0004', ts: '2026-06-10T18:30:00Z', type: 'bias_alert',           detail: 'Lithium variance: 12.3% > 10% threshold', actor: 'system',    hash: 'fe01…9a3c', sensitive: true },
  { id: 'ev-0003', ts: '2026-06-10T08:00:01Z', type: 'pipeline_complete',    detail: 'elapsed: 298.1s',                        actor: 'admin@gic',  hash: '22bd…7701', sensitive: true },
];

const MOCK_BIAS = [
  { commodity: 'Copper',       bias: 2.1,  mape: 7.0,  status: 'good' },
  { commodity: 'Steel',        bias: 4.8,  mape: 12.4, status: 'good' },
  { commodity: 'Lithium',      bias: 12.3, mape: 11.9, status: 'escalate' },
  { commodity: 'Aluminum',     bias: 6.2,  mape: 16.7, status: 'alert' },
  { commodity: 'Cobalt',       bias: 3.9,  mape: 14.7, status: 'good' },
  { commodity: 'Platinum',     bias: 2.4,  mape: 8.9,  status: 'good' },
  { commodity: 'Natural Gas',  bias: 18.7, mape: 31.1, status: 'escalate' },
  { commodity: 'Palladium',    bias: 14.1, mape: 29.1, status: 'escalate' },
];

const biasColor = (s) => ({ good: 'green', alert: 'yellow', escalate: 'red' }[s] || 'blue');
const biasLabel = (s) => ({ good: 'OK', alert: 'Alert', escalate: 'Escalate' }[s] || s);
const eventTypeColor = (t) => {
  if (t.includes('alert') || t.includes('escalation')) return 'red';
  if (t.includes('narrative') || t.includes('generated')) return 'blue';
  if (t.includes('trained') || t.includes('pipeline')) return 'green';
  return 'yellow';
};

export default function Governance() {
  const { user } = useAuth();
  const fullAudit = can(user, PERMISSIONS.VIEW_AUDIT_FULL);
  const canManageThresholds = can(user, PERMISSIONS.MANAGE_THRESHOLDS);

  const [selectedCommodity, setSelectedCommodity] = useState('Copper');
  const [narrative, setNarrative] = useState('');
  const [loading, setLoading] = useState(false);
  const [llmStatus, setLlmStatus] = useState(LLM_STATUS);
  const [sortKey, setSortKey] = useState('bias');
  const [sortDir, setSortDir] = useState('desc');

  useEffect(() => {
    gicApi.health().then((h) => {
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
      await new Promise((r) => setTimeout(r, 800));
      setNarrative(MOCK_NARRATIVES[selectedCommodity] || MOCK_NARRATIVES.Copper);
    }
    setLoading(false);
  };

  const sortedBias = useMemo(() => {
    const arr = [...MOCK_BIAS];
    const order = { good: 0, alert: 1, escalate: 2 };
    arr.sort((a, b) => {
      let av = a[sortKey];
      let bv = b[sortKey];
      if (sortKey === 'status') { av = order[a.status]; bv = order[b.status]; }
      if (sortKey === 'commodity') return sortDir === 'asc' ? a.commodity.localeCompare(b.commodity) : b.commodity.localeCompare(a.commodity);
      return sortDir === 'asc' ? av - bv : bv - av;
    });
    return arr;
  }, [sortKey, sortDir]);

  const setSort = (k) => {
    if (sortKey === k) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortKey(k); setSortDir('desc'); }
  };
  const sortArrow = (k) => (sortKey === k ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '');

  const auditShown = fullAudit ? MOCK_AUDIT : MOCK_AUDIT.filter((e) => !e.sensitive);

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
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <div className="rounded-xl p-5 border border-blue-700 lg:col-span-1" style={{ backgroundColor: 'rgba(59,130,246,0.08)' }}>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">🤖</span>
            <span className="text-slate-300 text-sm">LLM Backend</span>
            <Badge label={llmStatus.status === 'healthy' ? 'Active' : 'Offline'} color={llmStatus.status === 'healthy' ? 'green' : 'red'} />
          </div>
          <p className="text-white font-bold text-lg font-mono">{llmStatus.backend}</p>
          <p className="text-slate-400 text-xs mt-1 font-mono">{llmStatus.model}</p>
          <p className="text-slate-500 text-[11px] mt-1">
            {llmStatus.backend === 'ollama' ? 'Ollama llama3.2:1b (local)' : llmStatus.backend === 'transformers' ? 'HuggingFace · CPU · free, no API key' : 'Template fallback'}
          </p>
        </div>
        {[
          { label: 'Narrative Coverage', value: '12 / 12', detail: 'All commodities have narratives', color: 'green' },
          { label: 'Mean Confidence', value: '82%', detail: 'Self-reported ensemble confidence', color: 'blue' },
          { label: 'Audit Events', value: `${MOCK_AUDIT.length}`, detail: 'JSONL append-only · immutable', color: 'green' },
        ].map((card, i) => (
          <div key={i} className="rounded-xl p-5 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm">{card.label}</span>
              <Badge label="OK" color={card.color} />
            </div>
            <p className="text-white font-bold text-2xl">{card.value}</p>
            <p className="text-slate-500 text-xs mt-1">{card.detail}</p>
          </div>
        ))}
      </div>

      {/* LLM Narrative Generator */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-1">LLM Narrative Generator</h2>
        <p className="text-slate-500 text-xs mb-4">
          Powered by <code className="text-blue-400">{llmStatus.model}</code> ({llmStatus.backend}) · Falls back to Ollama llama3.2:1b if available
        </p>
        <div className="flex flex-wrap gap-3 mb-4">
          <select value={selectedCommodity} onChange={(e) => { setSelectedCommodity(e.target.value); setNarrative(''); }}
            className="rounded-lg px-3 py-2 border border-slate-600 text-slate-200 text-sm focus:outline-none focus:border-blue-500" style={{ backgroundColor: '#0f172a' }}>
            {COMMODITIES.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <LockedButton
            permission={PERMISSIONS.REGENERATE_NARRATIVES}
            onClick={generateNarrative}
            disabled={loading}
            lockedLabel="Regenerate narrative"
            lockHint="Regenerating LLM narratives requires Administrator access"
          >
            {loading ? '⏳ Generating…' : '🤖 Regenerate narrative'}
          </LockedButton>
        </div>

        {loading && <Loading message="Running LLM inference…" />}

        {narrative && !loading && (
          <div className="rounded-lg p-5 border border-blue-800 text-sm leading-relaxed" style={{ backgroundColor: 'rgba(59,130,246,0.07)' }}>
            <div className="flex items-center gap-2 mb-3">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              <span className="text-blue-300 text-xs font-medium">LLM narrative · {llmStatus.backend} backend · {selectedCommodity}</span>
            </div>
            <p className="text-slate-200">{narrative}</p>
          </div>
        )}

        {!narrative && !loading && (
          <div className="rounded-lg p-4 border border-slate-700 text-center text-slate-500 text-sm" style={{ backgroundColor: '#0f172a' }}>
            Select a commodity and regenerate the narrative (Admin), or view the latest stored narrative below.
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Audit Trail */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-100">Audit Trail (JSONL)</h2>
            <Badge label={fullAudit ? 'Full detail' : 'Summary view'} color={fullAudit ? 'green' : 'slate'} />
          </div>
          {!fullAudit && (
            <p className="text-xs text-slate-500 mb-3">
              🔒 Viewer mode — sensitive events and actor/hash fields are hidden. Full audit detail is Admin-only.
            </p>
          )}
          <div className="space-y-2 max-h-80 overflow-y-auto scrollbar-thin pr-1">
            {auditShown.map((ev, i) => (
              <div key={i} className="rounded-lg px-3 py-2 border border-slate-700 text-xs" style={{ backgroundColor: '#0f172a' }}>
                <div className="flex items-center justify-between mb-1">
                  <Badge label={ev.type.replace(/_/g, ' ')} color={eventTypeColor(ev.type)} />
                  <code className="text-slate-500">{ev.ts.replace('T', ' ').slice(0, 19)}</code>
                </div>
                <p className="text-slate-400">{ev.detail}</p>
                {fullAudit ? (
                  <div className="flex items-center justify-between mt-1">
                    <code className="text-slate-600">{ev.id} · {ev.actor}</code>
                    <code className="text-slate-600">sha256:{ev.hash}</code>
                  </div>
                ) : (
                  <code className="text-slate-600">{ev.id}</code>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Bias Tracking */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <div className="flex items-center justify-between mb-1">
            <h2 className="text-lg font-semibold text-slate-100">Forecast Bias Tracking</h2>
            <LockedButton
              permission={PERMISSIONS.MANAGE_THRESHOLDS}
              onClick={() => {}}
              lockedLabel="Edit thresholds"
              lockHint="Editing bias thresholds requires Administrator access"
              color="#475569" hoverColor="#334155"
              className="text-xs px-3 py-1.5"
            >
              ⚙ Edit thresholds
            </LockedButton>
          </div>
          <p className="text-slate-500 text-xs mb-4">
            &gt;5% → alert · &gt;10% → L6 governance escalation {canManageThresholds && <span className="text-blue-400">· thresholds editable</span>}
          </p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700 text-xs">
                <th className="text-left pb-2 cursor-pointer select-none hover:text-slate-200" onClick={() => setSort('commodity')}>Commodity{sortArrow('commodity')}</th>
                <th className="text-right pb-2 cursor-pointer select-none hover:text-slate-200" onClick={() => setSort('bias')}>Bias %{sortArrow('bias')}</th>
                <th className="text-right pb-2 cursor-pointer select-none hover:text-slate-200" onClick={() => setSort('mape')}>CV MAPE{sortArrow('mape')}</th>
                <th className="text-right pb-2 cursor-pointer select-none hover:text-slate-200" onClick={() => setSort('status')}>Status{sortArrow('status')}</th>
              </tr>
            </thead>
            <tbody>
              {sortedBias.map((b, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-2 text-slate-200">{b.commodity}</td>
                  <td className={`py-2 text-right font-mono ${b.bias > 10 ? 'text-red-400' : b.bias > 5 ? 'text-yellow-400' : 'text-green-400'}`}>{b.bias}%</td>
                  <td className="py-2 text-right text-slate-400">{b.mape}%</td>
                  <td className="py-2 text-right"><Badge label={biasLabel(b.status)} color={biasColor(b.status)} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
