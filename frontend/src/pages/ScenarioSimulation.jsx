import React, { useMemo, useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';
import Loading from '../components/common/Loading';
import LockedButton from '../components/common/LockedButton';
import DistributionHistogram from '../components/Charts/DistributionHistogram';
import TornadoChart from '../components/Charts/TornadoChart';
import FanChart from '../components/Charts/FanChart';
import { useAuth } from '../auth/AuthContext';
import { can, PERMISSIONS } from '../auth/permissions';
import { gicApi } from '../api/client';

// Preset EBIT values derived from the simulation formula:
//   meanEbit = BASE_EBIT × (1 + demand×0.9 − commodity×0.65 + fx×0.35)
// fx is SIGNED: positive fx (GBP weaker vs USD) → POSITIVE for a net UK exporter.
// marg = EBIT margin % = ebit / 19800 × 100  (Gross Margin is separately 35.9%).
// var95 = approximate 5th-percentile EBIT (£M) from fat-tail Monte Carlo.
const PRESETS = [
  // 1401×(1+0×0.9−0×0.65+0×0.35) = 1401
  { name: 'Base Case',        demand: 0,     commodity: 0,    fx: 0,    color: '#3b82f6', ebit: 1401, var95: 1039, marg: 7.1 },
  // 1401×(1+0.09+0.0325−0.007) = 1401×1.1155 ≈ 1563
  { name: 'Bull Market',      demand: 0.10,  commodity: -0.05, fx: -0.02, color: '#22c55e', ebit: 1563, var95: 1068, marg: 7.9 },
  // 1401×(1−0.072) = 1401×0.928 ≈ 1300
  { name: 'EU Demand -8%',    demand: -0.08, commodity: 0,    fx: 0,    color: '#f59e0b', ebit: 1300, var95: 856,  marg: 6.6 },
  // 1401×(1−0.045−0.26+0.035) = 1401×0.73 ≈ 1023
  { name: 'Commodity Crisis', demand: -0.05, commodity: 0.40, fx: 0.10, color: '#ef4444', ebit: 1023, var95: 114,  marg: 5.2 },
  // 1401×(1−0.0975) = 1401×0.9025 ≈ 1265
  { name: 'Lithium +15%',     demand: 0,     commodity: 0.15, fx: 0,    color: '#f97316', ebit: 1265, var95: 800,  marg: 6.4 },
  // 1401×(1+0.045+0.013) = 1401×1.058 ≈ 1482
  { name: 'Rate Cuts',        demand: 0.05,  commodity: -0.02, fx: 0,   color: '#8b5cf6', ebit: 1482, var95: 1104, marg: 7.5 },
  // 1401×(1−0.108−0.1625+0.028) = 1401×0.7575 ≈ 1061
  { name: 'Stagflation',      demand: -0.12, commodity: 0.25, fx: 0.08, color: '#dc2626', ebit: 1061, var95: 237,  marg: 5.4 },
];

const BASE_EBIT = 1401;
const SANDBOX_CAP = 1000;

const formatPct = (v) => `${v > 0 ? '+' : ''}${(v * 100).toFixed(0)}%`;

// Box–Muller normal sampler for the client-side mock Monte Carlo.
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Build a mock outcome distribution + summary stats from the shock vector.
// FX coefficient is SIGNED: positive fx (GBP weaker) helps a net UK exporter.
// Vol always uses absolute shocks since uncertainty is always positive.
function mockSimulate(demand, commodity, fx, n) {
  const meanEbit =
    BASE_EBIT * (1 + demand * 0.9 - commodity * 0.65 + fx * 0.35);
  const vol = BASE_EBIT * (0.14 + Math.abs(commodity) * 0.5 + Math.abs(demand) * 0.3 + Math.abs(fx) * 0.3);
  const samples = new Array(n);
  for (let i = 0; i < n; i++) {
    const tail = Math.random() < 0.06 ? randn() * 2.2 : 0;
    samples[i] = meanEbit + (randn() + tail) * vol;
  }
  samples.sort((a, b) => a - b);
  const pct = (p) => samples[Math.min(n - 1, Math.max(0, Math.floor(p * n)))];
  const var95 = pct(0.05);
  const tailSamples = samples.filter((s) => s <= var95);
  const cvar95 = tailSamples.length ? tailSamples.reduce((a, b) => a + b, 0) / tailSamples.length : var95;
  const mean = samples.reduce((a, b) => a + b, 0) / n;

  const lo = samples[0];
  const hi = samples[n - 1];
  const nb = 30;
  const w = (hi - lo) / nb || 1;
  const bins = Array.from({ length: nb }, (_, i) => ({ x: lo + w * (i + 0.5), count: 0 }));
  samples.forEach((s) => {
    const idx = Math.min(nb - 1, Math.max(0, Math.floor((s - lo) / w)));
    bins[idx].count += 1;
  });

  // EBIT margin = simulated mean EBIT / fixed revenue (£19,800M)
  const ebitMargin = (mean / 19800) * 100;

  return {
    stats: {
      operating_income: { mean: mean * 1e6, var_95: var95 * 1e6, cvar_95: cvar95 * 1e6, p25: pct(0.25) * 1e6, p75: pct(0.75) * 1e6 },
      ebit_margin: { mean: ebitMargin },
    },
    _bins: bins,
    _mean: mean,
    _var95: var95,
    _cvar95: cvar95,
  };
}

// Mock variance decomposition for offline display.
const MOCK_VAR_DECOMP = { commodity_pct: 62.4, demand_pct: 22.8, fx_pct: 14.8 };

// Mock monthly fan chart for offline display.
function buildMockFan() {
  const labels = [
    'Jul 26','Aug 26','Sep 26','Oct 26','Nov 26','Dec 26',
    'Jan 27','Feb 27','Mar 27','Apr 27','May 27','Jun 27',
  ];
  return labels.map((date, i) => {
    const base = 116 + i * 0.9;
    return {
      date,
      mean:  +(base).toFixed(1),
      p5:    +(base * 0.60).toFixed(1),
      p10:   +(base * 0.69).toFixed(1),
      p25:   +(base * 0.83).toFixed(1),
      p75:   +(base * 1.18).toFixed(1),
      p90:   +(base * 1.29).toFixed(1),
      p95:   +(base * 1.37).toFixed(1),
    };
  });
}

const MOCK_FAN = buildMockFan();

export default function ScenarioSimulation() {
  const { user } = useAuth();
  const canReal    = can(user, PERMISSIONS.RUN_SIMULATION);
  const canSandbox = can(user, PERMISSIONS.RUN_SANDBOX_SIMULATION);
  const canEdit    = can(user, PERMISSIONS.EDIT_SCENARIOS);

  const [selected, setSelected]   = useState('Base Case');
  const [demand, setDemand]       = useState(0);
  const [commodity, setCommodity] = useState(0);
  const [fx, setFx]               = useState(0);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [mode, setMode]           = useState(null);
  const [nSims, setNSims]         = useState(canReal ? 10000 : SANDBOX_CAP);

  // Variance decomposition — loaded on mount; falls back to mock offline
  const [varDecomp, setVarDecomp] = useState(null);
  const [varDecompLive, setVarDecompLive] = useState(false);

  // Monthly fan chart — loaded on mount; falls back to mock offline
  const [fanData, setFanData]   = useState(MOCK_FAN);
  const [fanLive, setFanLive]   = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const vd = await gicApi.varianceDecomposition();
        setVarDecomp(vd);
        setVarDecompLive(true);
      } catch {
        setVarDecomp(MOCK_VAR_DECOMP);
      }
    })();
    (async () => {
      try {
        const fan = await gicApi.monthlyFan();
        if (fan?.months?.length) {
          setFanData(fan.months);
          setFanLive(true);
        }
      } catch {
        setFanData(MOCK_FAN);
      }
    })();
  }, []);

  const preset = PRESETS.find((p) => p.name === selected) || PRESETS[0];

  const applyPreset = (p) => {
    setSelected(p.name);
    setDemand(Math.round(p.demand * 100));
    setCommodity(Math.round(p.commodity * 100));
    setFx(Math.round(p.fx * 100));
    setResult(null);
  };

  const run = async (runMode) => {
    setLoading(true);
    setResult(null);
    setMode(runMode);
    const effectiveN = runMode === 'sandbox' ? Math.min(nSims, SANDBOX_CAP) : nSims;

    // Local Monte Carlo always runs first — provides histogram bins and fallback stats.
    const local = mockSimulate(demand / 100, commodity / 100, fx / 100, effectiveN);

    if (runMode === 'real') {
      try {
        const res = await gicApi.runScenario({
          name: selected === 'Base Case' ? 'custom' : selected,
          demand_shock: demand / 100,
          commodity_shock: commodity / 100,
          n_simulations: effectiveN,
        });

        // Build merged result: start from local (for _bins/_mean/_var95/_cvar95),
        // then overlay backend simulation_stats for the KPI display cards.
        let merged = { ...local, _server: true };

        if (res.simulation_stats) {
          const sim = res.simulation_stats;
          // Backend stats are in raw £; gross_margin.mean is absolute £ so convert to %.
          const meanRevenue = sim.net_revenue?.mean || res.deterministic?.total_revenue || 0;
          const gmPct = meanRevenue > 0
            ? (sim.gross_margin?.mean || 0) / meanRevenue * 100
            : (res.deterministic?.margin_pct ?? local.stats.gross_margin.mean);

          merged.stats = {
            operating_income: {
              mean:   sim.operating_income?.mean   ?? local.stats.operating_income.mean,
              var_95: sim.operating_income?.var_95 ?? local.stats.operating_income.var_95,
              cvar_95: sim.operating_income?.cvar_95 ?? local.stats.operating_income.cvar_95,
              p25:    sim.operating_income?.p25    ?? local.stats.operating_income.p25,
              p75:    sim.operating_income?.p75    ?? local.stats.operating_income.p75,
            },
            ebit_margin: { mean: gmPct },
          };
          // Align histogram reference lines with backend stats (convert raw £ → £M)
          merged._mean  = (sim.operating_income?.mean   ?? local._mean  * 1e6) / 1e6;
          merged._var95 = (sim.operating_income?.var_95 ?? local._var95 * 1e6) / 1e6;
          merged._cvar95 = (sim.operating_income?.cvar_95 ?? local._cvar95 * 1e6) / 1e6;
        }

        // Use backend histogram bins when provided (x already in £M from server)
        if (res.histogram_bins?.length) {
          merged._bins = res.histogram_bins;
        }

        setResult(merged);
        setLoading(false);
        return;
      } catch {
        // Backend unreachable — fall through to local result
      }
    } else {
      await new Promise((r) => setTimeout(r, 500)); // perceived compute
    }
    setResult(local);
    setLoading(false);
  };

  const ebit     = result ? result.stats?.operating_income?.mean  / 1e6 : null;
  const var95    = result ? result.stats?.operating_income?.var_95 / 1e6 : null;
  const cvar95   = result ? result.stats?.operating_income?.cvar_95 / 1e6 : null;
  const margin   = result ? result.stats?.ebit_margin?.mean : null;
  const ebitDelta = ebit != null ? ebit - BASE_EBIT : null;

  const tornado = useMemo(() => {
    const d = demand / 100, c = commodity / 100, f = fx / 100;
    const dMag = Math.max(0.05, Math.abs(d));
    const cMag = Math.max(0.05, Math.abs(c));
    const fMag = Math.max(0.02, Math.abs(f));
    // FX is directional for a net UK exporter: GBP weaker (+fx) → +EBIT, GBP stronger (−fx) → −EBIT.
    // Tornado bars show the signed sensitivity range at current shock magnitude.
    return [
      { name: 'Demand',              low: -BASE_EBIT * 0.9  * dMag, high: BASE_EBIT * 0.9  * dMag },
      { name: 'Commodity',           low: -BASE_EBIT * 0.65 * cMag, high: BASE_EBIT * 0.65 * cMag },
      { name: 'FX (↑ GBP weaker)',  low: -BASE_EBIT * 0.35 * fMag, high: BASE_EBIT * 0.35 * fMag },
    ];
  }, [demand, commodity, fx]);

  const compareData = PRESETS.map((p) => ({
    name: p.name.replace('Commodity Crisis', 'Comm. Crisis').replace('Stagflation', 'Stagfl.'),
    ebit: p.ebit,
    fill: p.color,
  }));

  const maxSims = canReal ? 50000 : SANDBOX_CAP;

  const impactCard = (label, value, deltaGood, fmt) => (
    <div className="rounded-xl p-4 border" style={{ backgroundColor: '#0f172a', borderColor: deltaGood ? '#15803d' : '#b91c1c' }}>
      <p className="text-xs text-slate-400 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${deltaGood ? 'text-emerald-400' : 'text-red-400'}`}>{fmt(value)}</p>
    </div>
  );

  // Variance decomposition bar widths
  const vd = varDecomp || MOCK_VAR_DECOMP;
  const vdBars = [
    { label: 'Commodity', pct: vd.commodity_pct, color: '#f59e0b' },
    { label: 'Demand',    pct: vd.demand_pct,    color: '#3b82f6' },
    { label: 'FX',        pct: vd.fx_pct,        color: '#a78bfa' },
  ];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className="text-blue-400">ℹ️</span>
        Connect backend:{' '}
        <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code>
        {' '}— full simulation calls the real Monte Carlo engine when connected.
      </div>

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Scenario Simulation</h1>
          <p className="text-slate-400 text-sm mt-1">Monte Carlo · Fat-tail distributions (Student's t) · VaR / CVaR · FX positive = GBP weaker (net UK exporter benefit)</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${canReal ? 'border-emerald-700 text-emerald-300 bg-emerald-900/20' : 'border-slate-600 text-slate-400 bg-slate-700/30'}`}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: canReal ? '#34d399' : '#94a3b8' }} />
            {canReal ? 'Administrator · full-scale on actual data' : 'Viewer · sandbox on sample data'}
          </span>
        </div>
      </div>

      {/* ── Monte Carlo Engine Status ─────────────────────────────────────────── */}
      <div className="rounded-xl p-5 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <div className="flex flex-wrap items-center gap-6">
          <div className="flex items-center gap-3">
            <span className="text-lg">🎲</span>
            <div>
              <p className="text-sm font-semibold text-slate-200">Monte Carlo Engine</p>
              <p className="text-xs text-slate-500">Student's t fat-tails · NumPy vectorised · up to 50K sims</p>
            </div>
          </div>
          {[
            { label: 'Distribution',     value: 'Student\'s t (df=5)' },
            { label: 'Risk metrics',     value: 'VaR95 · CVaR95 · p5–p95' },
            { label: 'Shock sources',    value: 'Demand · Commodity · FX' },
            { label: 'Variance decomp',  value: varDecompLive ? '✓ Live' : 'Mock fallback', live: varDecompLive },
            { label: 'Fan chart',        value: fanLive ? '✓ Live' : 'Mock fallback', live: fanLive },
          ].map((m) => (
            <div key={m.label} className="text-xs">
              <p className="text-slate-500 uppercase tracking-wide">{m.label}</p>
              <p className={m.live === true ? 'text-emerald-400 font-semibold' : m.live === false ? 'text-amber-400' : 'text-slate-200'}>{m.value}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Scenario Presets */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold text-slate-100">Preset Scenarios</h2>
            {!canEdit && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-700/70 text-slate-400 border border-slate-600">🔒 Read-only</span>
            )}
          </div>
          <div className="space-y-2">
            {PRESETS.map((p) => (
              <button key={p.name} onClick={() => applyPreset(p)}
                className={`w-full text-left rounded-lg px-3 py-2.5 border transition-colors ${
                  selected === p.name
                    ? 'border-blue-500 bg-blue-900/30 text-white'
                    : 'border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                }`}
                style={selected !== p.name ? { backgroundColor: '#0f172a' } : {}}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{p.name}</span>
                  <span className={`text-xs ${p.ebit < 1000 ? 'text-red-400' : p.ebit > 1500 ? 'text-green-400' : 'text-yellow-400'}`}>
                    £{p.ebit}M EBIT
                  </span>
                </div>
                <div className="text-xs text-slate-500 mt-0.5">
                  D:{formatPct(p.demand)} · C:{formatPct(p.commodity)} · FX:{formatPct(p.fx)}
                </div>
              </button>
            ))}
          </div>
          <div className="mt-4">
            <LockedButton
              permission={PERMISSIONS.EDIT_SCENARIOS}
              onClick={() => {}}
              className="w-full"
              lockedLabel="Save scenario"
              lockHint="Editing or saving presets requires Administrator access"
            >
              💾 Save scenario
            </LockedButton>
          </div>
        </div>

        {/* What-If Builder */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-4">What-If Builder</h2>
          <div className="space-y-5">
            {[
              { label: 'Demand Shock',    val: demand,    set: setDemand,    min: -20, max: 20, color: '#3b82f6' },
              { label: 'Commodity Shock', val: commodity, set: setCommodity, min: -30, max: 50, color: '#f59e0b' },
              { label: 'FX Shock',        val: fx,        set: setFx,        min: -15, max: 15, color: '#a78bfa' },
            ].map(({ label, val, set, min, max, color }) => (
              <div key={label}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-400">{label}</span>
                  <span className={`font-bold ${val > 0 ? 'text-red-400' : val < 0 ? 'text-green-400' : 'text-slate-300'}`}>
                    {val > 0 ? '+' : ''}{val}%
                  </span>
                </div>
                <input type="range" min={min} max={max} step={1} value={val}
                  onChange={(e) => { set(Number(e.target.value)); setSelected('Custom'); setResult(null); }}
                  className="w-full" style={{ accentColor: color }}
                />
                <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                  <span>{min}%</span><span>0%</span><span>+{max}%</span>
                </div>
              </div>
            ))}

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">Simulations</span>
                <span className="text-slate-300">{nSims.toLocaleString()}{!canReal && ` / ${SANDBOX_CAP.toLocaleString()} cap`}</span>
              </div>
              <input type="range" min={1000} max={maxSims} step={1000} value={Math.min(nSims, maxSims)}
                onChange={(e) => setNSims(Number(e.target.value))}
                className="w-full" style={{ accentColor: '#22c55e' }} />
              <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                <span>1K</span><span>{canReal ? '25K' : '—'}</span><span>{canReal ? '50K' : '1K'}</span>
              </div>
            </div>
          </div>

          <div className="mt-5 space-y-2">
            {canReal ? (
              <button onClick={() => run('real')} disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white py-2.5 rounded-lg font-medium transition-colors">
                {loading && mode === 'real' ? '⏳ Running…' : `▶ Run on ACTUAL data (${nSims.toLocaleString()} sims)`}
              </button>
            ) : (
              <button disabled title="Full-scale simulation on actual data requires Administrator access"
                className="w-full inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium border border-slate-600 text-slate-400 cursor-not-allowed select-none"
                style={{ backgroundColor: 'rgba(100,116,139,0.12)' }}>
                <span aria-hidden>🔒</span> Run on ACTUAL data
                <span className="ml-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-slate-700/70 text-slate-400 border border-slate-600">Admin only</span>
              </button>
            )}

            {canSandbox && (
              <button onClick={() => run('sandbox')} disabled={loading}
                className="w-full bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-100 py-2.5 rounded-lg font-medium transition-colors border border-slate-600">
                {loading && mode === 'sandbox' ? '⏳ Running…' : `🧪 Run Sandbox (sample data · ${Math.min(nSims, SANDBOX_CAP).toLocaleString()} sims)`}
              </button>
            )}

            {!canReal && (
              <p className="text-xs text-slate-500 leading-relaxed">
                Sandbox runs use <span className="text-slate-300">sample data</span> capped at {SANDBOX_CAP.toLocaleString()} simulations.
                Sign in as <span className="text-blue-300 font-medium">Administrator</span> for full-scale simulation on actual data.
              </p>
            )}
          </div>
        </div>

        {/* Results / Impact cards */}
        <div className="space-y-4">
          {loading && (
            <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
              <Loading message={mode === 'sandbox' ? 'Running sandbox simulation…' : 'Running Monte Carlo engine…'} />
            </div>
          )}
          {result && !loading && (
            <>
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-200">Results</h3>
                <span className={`text-[10px] px-2 py-0.5 rounded-full border ${mode === 'sandbox' ? 'border-amber-700 text-amber-300 bg-amber-900/20' : 'border-emerald-700 text-emerald-300 bg-emerald-900/20'}`}>
                  {mode === 'sandbox' ? 'Sandbox (sample data)' : result._server ? 'Actual data · MC engine' : 'Actual data'}
                </span>
              </div>
              <div className="grid grid-cols-1 gap-3">
                {impactCard('Mean EBIT Δ vs Base', ebitDelta, ebitDelta >= 0, (v) => `${v >= 0 ? '+' : ''}£${Math.round(v).toLocaleString()}M`)}
                <div className="grid grid-cols-2 gap-3">
                  {impactCard('VaR (95%) — 5th pct EBIT', var95, false, (v) => `£${Math.round(v).toLocaleString()}M`)}
                  {impactCard('EBIT Margin', margin, margin >= 7, (v) => `${v.toFixed(1)}%`)}
                </div>
              </div>
              <div className="rounded-xl p-4 border border-slate-700 text-xs text-slate-400 space-y-1.5" style={{ backgroundColor: '#1e293b' }}>
                <div className="flex justify-between"><span>Mean EBIT</span><span className="text-white font-mono">£{Math.round(ebit).toLocaleString()}M</span></div>
                <div className="flex justify-between"><span>VaR 95%</span><span className="text-red-300 font-mono">£{Math.round(var95).toLocaleString()}M</span></div>
                <div className="flex justify-between"><span>CVaR 95% (expected shortfall)</span><span className="text-red-400 font-mono">£{Math.round(cvar95).toLocaleString()}M</span></div>
                {result._server && (
                  <div className="pt-1 border-t border-slate-700 text-emerald-500">
                    ✓ Statistics from backend Monte Carlo engine (NumPy, Student's t)
                  </div>
                )}
              </div>
            </>
          )}
          {!result && !loading && (
            <div className="rounded-xl p-6 border border-slate-700 text-center text-slate-500 text-sm" style={{ backgroundColor: '#1e293b' }}>
              Select a preset or adjust sliders, then run a simulation.
            </div>
          )}
        </div>
      </div>

      {/* ── Distribution + Tornado ────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-1">Outcome Distribution — EBIT (£M)</h2>
          <p className="text-slate-500 text-xs mb-4">
            Probability histogram with VaR95 / CVaR95 markers
            {result?._server && <span className="text-emerald-500 ml-2">· Bins from backend engine</span>}
          </p>
          {result && !loading ? (
            <DistributionHistogram bins={result._bins} mean={result._mean} var95={result._var95} cvar95={result._cvar95} color={preset.color} />
          ) : (
            <div className="h-64 flex items-center justify-center text-slate-600 text-sm">Run a simulation to see the outcome distribution.</div>
          )}
        </div>

        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-1">Sensitivity — Tornado (EBIT Δ£M)</h2>
          <p className="text-slate-500 text-xs mb-4">Marginal EBIT impact of each shock at current settings</p>
          <TornadoChart items={tornado} />
        </div>
      </div>

      {/* ── Variance Decomposition ────────────────────────────────────────────── */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <div className="flex items-center gap-3 mb-1">
          <h2 className="text-lg font-semibold text-slate-100">Variance Decomposition</h2>
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${varDecompLive ? 'border-emerald-700 text-emerald-400 bg-emerald-900/20' : 'border-amber-700 text-amber-400 bg-amber-900/20'}`}>
            {varDecompLive ? '✓ Live · backend MC' : 'Mock · connect backend'}
          </span>
        </div>
        <p className="text-slate-500 text-xs mb-5">
          Share of total P&L variance attributable to each risk source (3 × 3,000-simulation partial MC)
        </p>
        <div className="space-y-4">
          {vdBars.map((b) => (
            <div key={b.label}>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="text-slate-300 font-medium">{b.label}</span>
                <span className="font-mono font-bold" style={{ color: b.color }}>{b.pct.toFixed(1)}%</span>
              </div>
              <div className="h-4 rounded-full bg-slate-700/60 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{ width: `${Math.min(100, b.pct)}%`, backgroundColor: b.color, opacity: 0.85 }}
                />
              </div>
            </div>
          ))}
        </div>
        <p className="text-xs text-slate-500 mt-4 leading-relaxed">
          Commodity price movements account for the majority of P&L uncertainty —
          hedging lithium and steel exposures delivers the highest marginal reduction in financial risk.
          Demand and FX shocks are secondary but can amplify tail outcomes.
        </p>
      </div>

      {/* ── Monthly EBIT Fan Chart ────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-lg font-semibold text-white">Monthly EBIT Fan Chart</h2>
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${fanLive ? 'border-emerald-700 text-emerald-400 bg-emerald-900/20' : 'border-amber-700 text-amber-400 bg-amber-900/20'}`}>
            {fanLive ? '✓ Live · per-month MC' : 'Mock · connect backend'}
          </span>
        </div>
        <p className="text-slate-400 text-xs mb-3">
          Monthly operating income distribution — 90% and 50% confidence bands from Monte Carlo (2,000 sims per month)
        </p>
        <FanChart data={fanData} title="" color={preset.color} />
      </div>

      {/* ── Scenario Comparison ───────────────────────────────────────────────── */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">Scenario Comparison — EBIT (£M)</h2>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={compareData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
              formatter={(v) => [`£${v}M`, 'EBIT']}
            />
            <ReferenceLine y={BASE_EBIT} stroke="#64748b" strokeDasharray="4 2" label={{ value: 'Base', fill: '#64748b', fontSize: 11 }} />
            <Bar dataKey="ebit" radius={[4, 4, 0, 0]}>
              {compareData.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
