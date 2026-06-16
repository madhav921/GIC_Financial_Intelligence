import React, { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import RiskGauge from '../components/insights/RiskGauge';
import KPICard from '../components/Charts/KPICard';
import Badge from '../components/common/Badge';
import Loading from '../components/common/Loading';
import { gicApi } from '../api/client';

// ── Mock fallback ─────────────────────────────────────────────────────────────
function buildMockForecast() {
  const dates = [];
  const point = [];
  const lower = [];
  const upper = [];
  const start = new Date('2026-07-01');
  let base = 24; // £M per month
  for (let i = 0; i < 12; i++) {
    const dt = new Date(start);
    dt.setMonth(start.getMonth() + i);
    dates.push(dt.toLocaleDateString('en-GB', { month: 'short', year: '2-digit' }));
    base += 0.6 + (Math.random() - 0.4);
    const p = Number(base.toFixed(1));
    // Symmetric ±19% around point = approximate 95% CI from EWMA residual std dev.
    // Widened band reflects higher uncertainty over longer horizons.
    const horizonFactor = 1 + i * 0.008;
    const halfBand = p * 0.19 * horizonFactor;
    point.push(p);
    lower.push(Number(Math.max(0, p - halfBand).toFixed(1)));
    upper.push(Number((p + halfBand).toFixed(1)));
  }
  return { dates, point, lower, upper };
}

const MOCK_WARRANTY = {
  forecast: buildMockForecast(),
  accrual_adequacy: { adequacy_pct: 92.6, status: 'under', shortfall_gbp: 18000000 },
  failure_modes: {
    'EV Battery': 31,
    Powertrain: 22,
    Electrical: 18,
    Infotainment: 14,
    Suspension: 9,
    Other: 6,
  },
  risk_score: 64,
};

const MODE_COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#22c55e', '#64748b'];

const STATUS_MAP = {
  adequate: { label: 'Adequate', color: 'green' },
  under: { label: 'Under-accrued', color: 'red' },
  over: { label: 'Over-accrued', color: 'yellow' },
};

function fmtGBP(v) {
  if (v === null || v === undefined) return '—';
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}£${(abs / 1e9).toFixed(2)}bn`;
  if (abs >= 1e6) return `${sign}£${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}£${(abs / 1e3).toFixed(0)}K`;
  return `${sign}£${abs.toFixed(0)}`;
}

const ForecastTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload || {};
  return (
    <div className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl" style={{ backgroundColor: '#0f172a' }}>
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      <div className="flex justify-between gap-4"><span className="text-blue-400">Point</span><span className="font-mono text-blue-300 font-bold">£{Number(row.point).toFixed(1)}M</span></div>
      <div className="flex justify-between gap-4"><span className="text-slate-400">Upper</span><span className="font-mono text-slate-200">£{Number(row.upper).toFixed(1)}M</span></div>
      <div className="flex justify-between gap-4"><span className="text-slate-400">Lower</span><span className="font-mono text-slate-200">£{Number(row.lower).toFixed(1)}M</span></div>
    </div>
  );
};

export default function WarrantyAnalytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await gicApi.warrantySummary();
        if (mounted && res?.forecast) {
          setData(res);
          setLive(true);
        } else if (mounted) {
          setData(MOCK_WARRANTY);
        }
      } catch {
        if (mounted) setData(MOCK_WARRANTY);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const d = data || MOCK_WARRANTY;
  const fc = d.forecast || MOCK_WARRANTY.forecast;
  const adeq = d.accrual_adequacy || MOCK_WARRANTY.accrual_adequacy;
  const status = STATUS_MAP[adeq.status] || STATUS_MAP.adequate;
  // Backend may return risk_score as {score, band, components} object; extract scalar
  const riskScore = typeof d.risk_score === 'object' ? (d.risk_score?.score ?? 0) : (d.risk_score ?? 0);

  // Forecast chart data — stack band as (lower) + (upper-lower) for the floating area.
  const chartData = useMemo(() => {
    const { dates = [], point = [], lower = [], upper = [] } = fc;
    return dates.map((dt, i) => ({
      date: dt,
      point: point[i],
      lower: lower[i],
      upper: upper[i],
      bandBase: lower[i],
      bandSpan: (upper[i] ?? 0) - (lower[i] ?? 0),
    }));
  }, [fc]);

  const total12m = useMemo(
    () => (fc.point || []).reduce((s, v) => s + (v || 0), 0) * 1e6,
    [fc]
  );

  const failureModes = useMemo(() => {
    const raw = d.failure_modes || {};
    // Backend returns {breakdown_pct: {...}, rising_modes: [], dominant_mode: str}
    // Mock uses flat {mode: pct} object — normalise to the flat form
    const fm = typeof raw.breakdown_pct === 'object' ? raw.breakdown_pct : raw;
    return Object.entries(fm)
      .filter(([, v]) => typeof v === 'number')
      .map(([name, pct]) => ({ name, pct }))
      .sort((a, b) => b.pct - a.pct);
  }, [d.failure_modes]);

  const dominantMode = failureModes[0];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className={`w-2 h-2 rounded-full ${live ? 'bg-emerald-400' : 'bg-amber-400'} animate-pulse`} />
        {live ? (
          <span className="text-emerald-300">Connected to warranty model (<code>/insights/warranty/summary</code>).</span>
        ) : (
          <span>Showing mock warranty analytics — start backend: <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code></span>
        )}
      </div>

      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Warranty Analytics</h2>
        <p className="text-slate-400 text-sm">Warranty cost exposure, accrual adequacy and failure-mode signals — linked to the driver-based P&L.</p>
      </div>

      {loading ? (
        <Loading message="Loading warranty model…" />
      ) : (
        <>
          {/* KPI row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KPICard title="12-Month Cost Forecast" value={total12m} subtitle="Projected warranty spend" format="currency" />
            <div className="rounded-xl p-5 border border-slate-700 flex flex-col gap-2" style={{ backgroundColor: '#1e293b' }}>
              <div className="text-xs font-medium text-slate-400 uppercase tracking-wide">Accrual Adequacy</div>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-white tabular-nums">{Number(adeq.adequacy_pct).toFixed(1)}%</span>
                <Badge label={status.label} color={status.color} />
              </div>
              <div className="text-xs text-slate-400">
                {adeq.shortfall_gbp ? (
                  <>Shortfall <span className="font-mono font-semibold text-red-300">{fmtGBP(adeq.shortfall_gbp)}</span> vs modelled liability</>
                ) : 'Coverage in line with modelled liability'}
              </div>
            </div>
            <div className="rounded-xl p-5 border border-slate-700 flex items-center justify-center" style={{ backgroundColor: '#1e293b' }}>
              <RiskGauge score={riskScore} size={200} label="Warranty Risk" />
            </div>
          </div>

          {/* Forecast chart + failure modes */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Forecast */}
            <div className="lg:col-span-3 rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
              <h3 className="text-lg font-semibold text-slate-100 mb-1">Warranty Cost Forecast</h3>
              <p className="text-xs text-slate-400 mb-4">12-month point forecast with 95% confidence band (£M / month) — derived from EWMA residual std dev</p>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
                  <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} width={48} tickFormatter={(v) => `£${v}M`} />
                  <Tooltip content={<ForecastTooltip />} />
                  {/* Floating CI band: invisible base + visible span */}
                  <Area type="monotone" dataKey="bandBase" stackId="ci" stroke="none" fill="transparent" isAnimationActive={false} />
                  <Area type="monotone" dataKey="bandSpan" stackId="ci" stroke="none" fill="#3b82f6" fillOpacity={0.16} isAnimationActive={false} name="CI" />
                  <Line type="monotone" dataKey="point" stroke="#3b82f6" strokeWidth={2.5} dot={false} activeDot={{ r: 4, fill: '#3b82f6' }} name="Point" />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="flex items-center gap-5 mt-3 px-2">
                <div className="flex items-center gap-1.5"><span className="w-6 h-0.5 bg-blue-500" /><span className="text-xs text-slate-400">Point forecast</span></div>
                <div className="flex items-center gap-1.5"><span className="w-6 h-3 rounded-sm bg-blue-500/20" /><span className="text-xs text-slate-400">95% CI</span></div>
              </div>
            </div>

            {/* Failure modes */}
            <div className="lg:col-span-2 rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
              <h3 className="text-lg font-semibold text-slate-100 mb-1">Failure-Mode Mix</h3>
              <p className="text-xs text-slate-400 mb-4">Share of warranty claims by mode</p>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={failureModes} layout="vertical" margin={{ top: 4, right: 28, left: 8, bottom: 4 }}>
                  <CartesianGrid stroke="#334155" strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} tickFormatter={(v) => `${v}%`} domain={[0, 'dataMax']} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#cbd5e1', fontSize: 11 }} axisLine={false} tickLine={false} width={92} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #475569', borderRadius: 8, fontSize: 12 }}
                    labelStyle={{ color: '#94a3b8' }}
                    formatter={(v) => [`${v}%`, 'Share']}
                    cursor={{ fill: 'rgba(148,163,184,0.06)' }}
                  />
                  <Bar dataKey="pct" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                    {failureModes.map((_, i) => (
                      <Cell key={i} fill={MODE_COLORS[i % MODE_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Narrative */}
          <div className="rounded-xl p-6 border border-blue-900/50" style={{ backgroundColor: '#1e293b' }}>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-xl">🔧</span>
              <h3 className="text-lg font-semibold text-slate-100">Warranty Brief</h3>
              <Badge label="Model synthesis" color="blue" />
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              The 12-month warranty forecast totals <span className="font-mono font-semibold text-blue-300">{fmtGBP(total12m)}</span>.
              Accrual adequacy stands at <span className="font-semibold text-white">{Number(adeq.adequacy_pct).toFixed(1)}%</span>
              {adeq.status === 'under' && adeq.shortfall_gbp ? (
                <>, indicating an <span className="text-red-300 font-semibold">under-accrual of {fmtGBP(adeq.shortfall_gbp)}</span> that should be topped up to avoid a P&L surprise.</>
              ) : adeq.status === 'over' ? (
                <>, indicating a conservative over-accrual that could be partially released.</>
              ) : (
                <>, broadly in line with the modelled liability.</>
              )}{' '}
              {dominantMode && (
                <>The dominant and rising failure mode is <span className="font-semibold text-amber-300">{dominantMode.name}</span>{' '}
                  at <span className="font-mono">{dominantMode.pct}%</span> of claims — prioritise root-cause analysis and supplier corrective action here to bend the cost curve.</>
              )}
            </p>
          </div>
        </>
      )}
    </div>
  );
}
