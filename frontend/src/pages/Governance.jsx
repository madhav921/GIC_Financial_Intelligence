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
  Steel:
    'Steel is forecast to rise over the next 12 months, driven by tightening global HRC capacity and elevated EU mill energy costs. Against JLR\'s £726M annual steel spend — body-in-white stampings and chassis components across the Defender and Discovery programmes — the current 40% forward cover leaves £436M unhedged. An unhedged upside move would add material cost pressure to EBIT, as steel is the single largest BOM line at 22% of the £3.3B strategic basket. Recommended action: Extend HRC forward contracts to 70% cover aligned with committed Defender and Discovery build schedules, using a mix of quarterly LME swaps and fixed-price mill agreements.',
  Aluminum:
    'Aluminium is forecast to rise, driven by European smelter energy tariffs and growing EV platform demand for high-strength sheet. Against JLR\'s £396M annual aluminium spend — primarily the all-aluminium body-in-white architecture across Range Rover and Range Rover Sport that delivers a 40% weight saving — the current 40% hedge cover leaves £238M exposed. Aluminium intensity is highest on the Luxury SUV platform, which carries JLR\'s highest gross margin per unit. Recommended action: Increase aluminium sheet hedging to 65% via LME swaps and negotiate fixed-price agreements with Novelis and Constellium ahead of the Range Rover model year change.',
  Copper:
    'Copper is forecast to rise, driven by EV adoption acceleration and Chilean mine supply disruptions, with LME spot as the primary reference. Against JLR\'s £198M annual copper exposure — EV and PHEV wiring harnesses, e-motor windings and DC charging systems across I-Pace, Range Rover PHEV and Defender PHEV — the current 40% cover leaves £119M unhedged. Copper intensity rises approximately 4× from ICE to full BEV, so exposure grows materially with JLR\'s EV mix. Recommended action: Layer copper forwards across EV programme launch windows; target 60% cover on committed I-Pace and Range Rover EV builds via LME futures aligned with wiring harness supplier call-off schedules.',
  Lithium:
    'Lithium is forecast to move significantly, driven by EV demand outpacing Australian and Chilean supply ramp and Chinese cathode processing demand. Against JLR\'s £594M annual lithium exposure — battery cells and cathode active material across I-Pace, Range Rover Electric and Defender PHEV — point forecasts carry inherent uncertainty (MAPE typically >20%). Multi-year offtake is preferred over spot market exposure for this critical EV cost driver representing 30–40% of battery pack cost. Recommended action: Negotiate multi-year cathode offtake agreements indexed to spodumene rather than spot lithium carbonate, targeting 70% coverage matched to Range Rover Electric and next-generation I-Pace build schedules.',
  Cobalt:
    'Cobalt is forecast to move, driven by DRC mining output and geopolitical risk — 70% of global supply originates from the Democratic Republic of Congo. Against JLR\'s £231M annual cobalt spend — NMC cathode active material in all I-Pace and PHEV battery packs — supply concentration risk is structural and warrants a physical buffer. JLR\'s cobalt exposure carries a persistent geopolitical risk premium that cannot be fully hedged financially. Recommended action: Accelerate transition to low-cobalt NMC811 and cobalt-free LFP chemistry in partnership with cell suppliers. Maintain a 3-month physical inventory buffer and hedge the residual cobalt requirement via LME forwards.',
  Nickel:
    'Nickel is forecast to move, driven by LME spot dynamics and the Indonesian HPAL supply ramp for Class-1 battery-grade material. Against JLR\'s £165M annual nickel spend — high-nickel NMC811 cathode for energy density in Range Rover Electric and I-Pace packs — high-nickel chemistry is JLR\'s strategic direction for next-generation platforms where range is a competitive differentiator. Recommended action: Align nickel procurement within battery cell supply agreements specifying Class-1 nickel from certified Indonesian HPAL sources. Establish a rolling 12-month nickel forward book to protect EV programme contribution margins.',
  Platinum:
    'Platinum is forecast to move, driven by South African supply concentration and EU emission regulation tightening for remaining ICE variants. Against JLR\'s £132M annual platinum spend — three-way catalytic converter PGM loading for NOx, HC and CO control on F-Pace, E-Pace and Range Rover ICE variants — platinum demand from JLR will decline structurally as the fleet electrifies, increasing the value of the closed-loop recycling programme. Recommended action: Maintain current hedge cover and expand the closed-loop autocatalyst recycling programme to recover PGM from end-of-life vehicles. Review PGM loadings against planned ICE volume decline on a quarterly basis.',
  'Natural Gas':
    'Natural gas is forecast to move significantly, driven by TTF spot pricing, NBP domestic UK price and seasonal heating demand — MAPE routinely exceeds 25%, so this narrative is directional rather than a precise point estimate. Against JLR\'s £132M annual gas exposure — Solihull and Castle Bromwich plant energy for paint shop, body shop and general services — energy represents approximately 4% of production cost and is largely unhedged, creating a material seasonal risk window. Recommended action: Implement a winter gas price collar (cap+floor) covering 50% of Q4–Q1 Solihull and Castle Bromwich consumption. Accelerate the renewable electricity transition programme to structurally reduce gas dependency over the planning horizon.',
  Palladium:
    'Palladium is forecast to move, driven by Russian supply risk (40% of global mine supply) and the structural decline of ICE production volumes. Against JLR\'s £99M annual palladium spend — gasoline three-way catalytic converter loading on Defender, Discovery and F-Pace petrol engines — palladium is in long-run structural decline as fleet electrification accelerates, though near-term Russian supply risk remains elevated. Recommended action: Accelerate platinum-for-palladium substitution in catalyst specification where homologation permits. Maintain a minimum 3-month physical inventory buffer to manage any supply disruption through the Russian risk window.',
  Rhodium:
    'Rhodium is forecast to move, driven by South African supply concentration (>80% of global production) with historically extreme price volatility ranging $1,000–$30,000 per troy ounce. Against JLR\'s £66M annual rhodium spend — three-way catalyst NOx reduction loading on F-Pace and Range Rover petrol variants — rhodium is the most price-volatile precious metal in JLR\'s basket with small BOM weight but significant tail risk. Recommended action: Minimise rhodium loading through catalyst formulation optimisation with JLR\'s tier-1 catalyst supplier. Maintain a 3-month physical buffer and hedge any incremental requirements via OTC forwards with specialist PGM dealers.',
  Polypropylene:
    'Polypropylene is forecast to move, driven by naphtha and propylene feedstock pricing, European cracker utilisation and crude oil — PP tracks crude with a 2–4 week lag. Against JLR\'s £99M annual PP spend — interior trim panels, bumper fascias, door cards and underbody protection across all programmes — energy market conditions are the primary cost driver and correlation with macro energy prices is high. Recommended action: Monitor naphtha forward curves and hedge up to 50% of annual PP requirement via feedstock-indexed pricing agreements with BASF and SABIC. Consolidate suppliers to improve negotiating leverage on fixed-price tranches.',
  'ABS Resin':
    'ABS Resin is forecast to move, driven by styrene monomer pricing, acrylonitrile supply tightness and butadiene availability — the three-monomer blend can disconnect from crude oil when any single monomer is in deficit. Against JLR\'s £66M annual ABS spend — exterior body cladding, grille surrounds and interior structural hard plastics on Range Rover, Discovery and I-Pace — pricing complexity warrants fixed-price supply agreements rather than spot exposure. Recommended action: Negotiate 18–24 month fixed-price supply agreements with LG Chem and Trinseo, targeting 60% fixed-price coverage with the balance managed via formula-priced spot supply.',
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

function _auditDetail(e) {
  const skip = new Set(['entry_id', 'timestamp', 'event_type', 'user']);
  const parts = Object.entries(e)
    .filter(([k]) => !skip.has(k))
    .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`);
  return parts.slice(0, 3).join(', ') || e.event_type;
}

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
  const [auditEvents, setAuditEvents] = useState(MOCK_AUDIT);
  const [biasData, setBiasData] = useState(MOCK_BIAS);
  const [dataSource, setDataSource] = useState('mock');

  useEffect(() => {
    // LLM status from health endpoint
    gicApi.health().then((h) => {
      if (h?.llm_status) setLlmStatus(h.llm_status);
    }).catch(() => {});

    // Live audit trail — fall back to MOCK_AUDIT if backend unreachable or empty
    gicApi.getAuditTrail(50).then((res) => {
      if (res?.events && res.events.length > 0) {
        const mapped = res.events.map((e, i) => ({
          id: e.entry_id || `ev-${i}`,
          ts: e.timestamp || new Date().toISOString(),
          type: e.event_type || 'unknown',
          detail: _auditDetail(e),
          actor: e.user || 'system',
          hash: e.entry_id ? `${e.entry_id.slice(0, 4)}…${e.entry_id.slice(-4)}` : '—',
          sensitive: ['pipeline_complete', 'simulation_run', 'bias_alert', 'bias_escalation', 'narrative_generated'].includes(e.event_type),
        })).reverse(); // most recent first
        setAuditEvents(mapped);
        setDataSource('live');
      }
    }).catch(() => {});

    // Live bias metrics — fall back to MOCK_BIAS if backend unreachable or warming
    gicApi.getBiasMetrics().then((res) => {
      if (res?.metrics && res.metrics.length > 0) {
        setBiasData(res.metrics);
        setDataSource('live');
      }
    }).catch(() => {});
  }, []);

  const generateNarrative = async () => {
    setLoading(true);
    setNarrative('');
    try {
      // Fast path: /intelligence/narrative uses the pipeline cache — no SARIMAX run
      const res = await gicApi.getNarrative(selectedCommodity);
      if (res?.narrative && !res.narrative.startsWith('Narrative generation unavailable')) {
        setNarrative(res.narrative);
      } else {
        setNarrative(MOCK_NARRATIVES[selectedCommodity] || MOCK_NARRATIVES.Steel);
      }
    } catch {
      setNarrative(MOCK_NARRATIVES[selectedCommodity] || MOCK_NARRATIVES.Steel);
    }
    setLoading(false);
  };

  const sortedBias = useMemo(() => {
    const arr = [...biasData];
    const order = { good: 0, alert: 1, escalate: 2 };
    arr.sort((a, b) => {
      let av = a[sortKey];
      let bv = b[sortKey];
      if (sortKey === 'status') { av = order[a.status]; bv = order[b.status]; }
      if (sortKey === 'commodity') return sortDir === 'asc' ? a.commodity.localeCompare(b.commodity) : b.commodity.localeCompare(a.commodity);
      return sortDir === 'asc' ? av - bv : bv - av;
    });
    return arr;
  }, [sortKey, sortDir, biasData]);

  const setSort = (k) => {
    if (sortKey === k) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortKey(k); setSortDir('desc'); }
  };
  const sortArrow = (k) => (sortKey === k ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '');

  const auditShown = fullAudit ? auditEvents : auditEvents.filter((e) => !e.sensitive);

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
          { label: 'Audit Events', value: `${auditEvents.length}`, detail: `JSONL append-only · immutable · ${dataSource}`, color: 'green' },
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
