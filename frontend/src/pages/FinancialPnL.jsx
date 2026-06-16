import React, { useMemo, useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, ReferenceLine,
} from 'recharts';
import WaterfallChart from '../components/Charts/WaterfallChart';
import KPICard from '../components/Charts/KPICard';
import LockedButton from '../components/common/LockedButton';
import { PERMISSIONS } from '../auth/permissions';
import { gicApi } from '../api/client';

// USD → GBP conversion. Backend serves P&L in USD (config uses avg_price_usd).
const USD_TO_GBP = 1 / 1.27;

// Full P&L walk: Revenue → Gross Margin → EBIT → Net Income
// Gross Margin = Revenue − Material COGS = 19800 − 12700 = 7100 (35.9%)
// EBIT = 7100 − 495 − 1140 − 4064 = 1401 (7.1%)
// Net Income = (1401 − 180) × (1 − 0.21) = 1221 × 0.79 = 965 (4.9%)
const STATIC_WATERFALL = [
  { label: 'Net Revenue',        value: 19800, type: 'total' },
  { label: 'Material COGS',      value: -12700, type: 'negative' },
  { label: 'Gross Margin',       value: 7100,  type: 'total' },
  { label: 'Warranty',           value: -495,  type: 'negative' },
  { label: 'Depreciation',       value: -1140, type: 'negative' },
  { label: 'Other OpEx',         value: -4064, type: 'negative' },
  { label: 'EBIT',               value: 1401,  type: 'total' },
  { label: 'Net Finance Costs',  value: -180,  type: 'negative' },
  { label: 'Pre-tax Profit',     value: 1221,  type: 'total' },
  { label: 'Tax (21%)',          value: -256,  type: 'negative' },
  { label: 'Net Income',         value: 965,   type: 'total' },
];

const STATIC_BASE_EBIT = 1401; // £M fallback

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function buildMonthly(annualRevGbpM, annualEbitGbpM) {
  return MONTHS.map((m, i) => {
    const seasonal = 1 + 0.12 * Math.sin((i / 12) * Math.PI * 2);
    const revenue = Math.round((annualRevGbpM / 12) * seasonal);
    const margin = +(35.9 + Math.sin((i / 12) * Math.PI * 2 - 0.5) * 1.6).toFixed(1);
    const quarterly = 0.96 + (i % 3) * 0.04;
    const ebit = Math.round((annualEbitGbpM / 12) * seasonal * quarterly);
    return { month: m, revenue, margin, ebit };
  });
}

const COMMODITY_BASKET_GBP = 3300; // £M tracked strategic commodity spend
const SENSITIVITY = [
  { commodity: 'Steel',    bomWeight: 22, impact1pct: -Math.round(0.22 * COMMODITY_BASKET_GBP * 0.01 * 10) / 10 },
  { commodity: 'Lithium',  bomWeight: 18, impact1pct: -Math.round(0.18 * COMMODITY_BASKET_GBP * 0.01 * 10) / 10 },
  { commodity: 'Aluminum', bomWeight: 12, impact1pct: -Math.round(0.12 * COMMODITY_BASKET_GBP * 0.01 * 10) / 10 },
  { commodity: 'Cobalt',   bomWeight: 7,  impact1pct: -Math.round(0.07 * COMMODITY_BASKET_GBP * 0.01 * 10) / 10 },
  { commodity: 'Copper',   bomWeight: 6,  impact1pct: -Math.round(0.06 * COMMODITY_BASKET_GBP * 0.01 * 10) / 10 },
  { commodity: 'Nickel',   bomWeight: 5,  impact1pct: -Math.round(0.05 * COMMODITY_BASKET_GBP * 0.01 * 10) / 10 },
];

const STATIC_SEGMENTS = [
  { segment: 'Luxury SUV',  revenue: 8400, volume: 80000,  margin: 22.1, cogs: 6540, color: '#3b82f6' },
  { segment: 'Premium SUV', revenue: 6640, volume: 120000, margin: 17.4, cogs: 5484, color: '#22c55e' },
  { segment: 'Performance', revenue: 2960, volume: 65000,  margin: 15.8, cogs: 2493, color: '#f59e0b' },
  { segment: 'EV',          revenue: 1800, volume: 45000,  margin: 12.3, cogs: 1579, color: '#a78bfa' },
];

const TrendTip = ({ active, payload, label, metric }) => {
  if (!active || !payload?.length) return null;
  const v = payload[0].value;
  return (
    <div className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl" style={{ backgroundColor: '#0f172a' }}>
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      <p className="text-white font-mono">{metric === 'margin' ? `${v}%` : `£${v.toLocaleString()}M`}</p>
    </div>
  );
};

export default function FinancialPnL() {
  const [shockCommodity, setShockCommodity] = useState('Steel');
  const [shockPct, setShockPct] = useState(0);
  const [trendMetric, setTrendMetric] = useState('revenue');

  // Live P&L state from backend
  const [liveKpis, setLiveKpis] = useState(null);
  const [kpiSource, setKpiSource] = useState('loading');

  useEffect(() => {
    gicApi.getAnnualPnL()
      .then((data) => {
        // Backend returns values in USD; convert to GBP £M for display.
        // total_revenue in USD / 1.27 / 1e6 = £M
        const revGbpM = Math.round(data.total_revenue * USD_TO_GBP / 1e6);
        const grossPct = data.gross_margin_pct || 35.9;
        const grossGbpM = Math.round(revGbpM * grossPct / 100);
        const ebitGbpM = Math.round(data.ebit * USD_TO_GBP / 1e6);
        const netGbpM = Math.round(data.net_income * USD_TO_GBP / 1e6);
        const ebitMarginPct = ((ebitGbpM / revGbpM) * 100).toFixed(1);
        setLiveKpis({
          revGbpM,
          grossPct: grossPct.toFixed(1),
          grossGbpM,
          ebitGbpM,
          netGbpM,
          ebitMarginPct,
          cogsRevPct: (100 - grossPct).toFixed(1),
        });
        setKpiSource('live');
      })
      .catch(() => {
        setKpiSource('mock');
      });
  }, []);

  const baseEbit = liveKpis?.ebitGbpM ?? STATIC_BASE_EBIT;
  const baseRev = liveKpis?.revGbpM ?? 19800;
  const MONTHLY = useMemo(() => buildMonthly(baseRev, baseEbit), [baseRev, baseEbit]);

  const sel = SENSITIVITY.find((s) => s.commodity === shockCommodity) || SENSITIVITY[0];
  const ebitDelta = useMemo(() => Math.round(sel.impact1pct * shockPct), [sel, shockPct]);
  const shockedEBIT = baseEbit + ebitDelta;

  const trendMeta = {
    revenue: { label: 'Revenue (£M)', color: '#3b82f6' },
    margin: { label: 'Gross Margin (%)', color: '#22c55e' },
    ebit: { label: 'EBIT (£M)', color: '#a78bfa' },
  }[trendMetric];

  const isLive = kpiSource === 'live';

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Data source banner */}
      <div className={`rounded-lg px-4 py-3 text-xs border flex items-center gap-2 ${isLive ? 'border-green-700 text-green-300 bg-green-900/15' : 'text-slate-400 border-slate-700'}`} style={!isLive ? { backgroundColor: '#1e293b' } : {}}>
        {isLive ? (
          <>
            <span className="text-green-400 font-bold">●</span>
            <span>Live P&L — KPI strip loaded from backend pipeline (data/raw/ → DataLoader → FinancialLayerController)</span>
            <span className="ml-auto text-green-600">Waterfall & segments use JLR-calibrated static structure</span>
          </>
        ) : kpiSource === 'loading' ? (
          <>
            <span className="text-blue-400 animate-pulse">↻</span>
            <span>Loading P&L from backend...</span>
          </>
        ) : (
          <>
            <span className="text-yellow-400">ℹ️</span>
            <span>Backend offline — showing JLR-calibrated mock data. Run <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code> for live P&L.</span>
          </>
        )}
      </div>

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Financial P&amp;L</h1>
          <p className="text-slate-400 text-sm mt-1">EBIT waterfall · segment contribution · monthly trend · commodity sensitivity</p>
        </div>
        <LockedButton
          permission={PERMISSIONS.EXPORT_REPORTS}
          onClick={() => {}}
          lockedLabel="Export P&L"
          lockHint="Exporting the P&L report requires Administrator access"
        >
          ⬇ Export P&amp;L
        </LockedButton>
      </div>

      {/* KPIs — prefer live backend data */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title={`Net Revenue${isLive ? ' ●' : ''}`}
          value={isLive ? `£${(liveKpis.revGbpM / 1000).toFixed(1)}B` : '£19.8B'}
          subtitle="Annual"
          change="+3.2%"
          changeType="up"
        />
        <KPICard
          title={`Gross Margin${isLive ? ' ●' : ''}`}
          value={isLive ? `£${(liveKpis.grossGbpM / 1000).toFixed(1)}B` : '£7.1B'}
          subtitle={isLive ? `${liveKpis.grossPct}%` : '35.9%'}
          change="-0.6pp"
          changeType="down"
        />
        <KPICard
          title={`EBIT${isLive ? ' ●' : ''}`}
          value={isLive ? `£${liveKpis.ebitGbpM.toLocaleString()}M` : '£1,401M'}
          subtitle={isLive ? `${liveKpis.ebitMarginPct}% margin` : '7.1% margin'}
          change="+8.3%"
          changeType="up"
        />
        <KPICard
          title={`COGS / Revenue${isLive ? ' ●' : ''}`}
          value={isLive ? `${liveKpis.cogsRevPct}%` : '64.1%'}
          subtitle="Commodity basket £3.3B of COGS"
          change="+0.4pp"
          changeType="down"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Waterfall — static JLR-calibrated P&L walk */}
        <WaterfallChart data={STATIC_WATERFALL} />

        {/* Monthly trend */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-100">Monthly Trend</h2>
            <div className="flex gap-1.5">
              {[
                { k: 'revenue', label: 'Revenue' },
                { k: 'margin', label: 'Margin' },
                { k: 'ebit', label: 'EBIT' },
              ].map((t) => (
                <button key={t.k} onClick={() => setTrendMetric(t.k)}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium border transition-colors ${
                    trendMetric === t.k ? 'bg-blue-600/30 border-blue-500 text-blue-200' : 'border-slate-700 text-slate-500 hover:text-slate-300'
                  }`}>
                  {t.label}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={MONTHLY} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} width={48} />
              <Tooltip content={<TrendTip metric={trendMetric} />} />
              <Line type="monotone" dataKey={trendMetric} stroke={trendMeta.color} strokeWidth={2.5} dot={{ r: 2, fill: trendMeta.color }} activeDot={{ r: 5 }} name={trendMeta.label} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Segment contribution */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">Segment Contribution — Revenue &amp; Contribution Margins (£M)</h2>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={STATIC_SEGMENTS} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="segment" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => `£${v}M`} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #475569', borderRadius: 8 }}
              formatter={(v, n, p) => [`£${v.toLocaleString()}M · ${p.payload.margin}% contribution margin*`, p.payload.segment]}
            />
            <Bar dataKey="revenue" radius={[4, 4, 0, 0]}>
              {STATIC_SEGMENTS.map((s, i) => <Cell key={i} fill={s.color} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <table className="w-full text-sm mt-4">
          <thead>
            <tr className="text-slate-400 border-b border-slate-700">
              <th className="text-left pb-2">Segment</th>
              <th className="text-right pb-2">Revenue (£M)</th>
              <th className="text-right pb-2">COGS (£M)</th>
              <th className="text-right pb-2">Volume</th>
              <th className="text-right pb-2">Contribution Margin %*</th>
            </tr>
          </thead>
          <tbody>
            {STATIC_SEGMENTS.map((s, i) => (
              <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/50">
                <td className="py-2 text-slate-200 font-medium">
                  <span className="inline-block w-2 h-2 rounded-sm mr-2" style={{ backgroundColor: s.color }} />{s.segment}
                </td>
                <td className="py-2 text-right text-slate-200">£{s.revenue.toLocaleString()}M</td>
                <td className="py-2 text-right text-slate-300">£{s.cogs.toLocaleString()}M</td>
                <td className="py-2 text-right text-slate-300">{s.volume.toLocaleString()}</td>
                <td className={`py-2 text-right font-medium ${s.margin >= 20 ? 'text-green-400' : s.margin >= 15 ? 'text-yellow-400' : 'text-red-400'}`}>{s.margin}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-[11px] text-slate-500 mt-3">
          * Contribution margin after all costs allocated to each segment (material, warranty, depreciation, overhead).
          Company-level Gross Margin uses Material COGS only and is higher because unallocated overhead sits above segment level.
        </p>
      </div>

      {/* Commodity Sensitivity — interactive */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-100">Commodity Cost Sensitivity</h2>
            <p className="text-slate-400 text-xs mt-1">
              EBIT impact of a single-commodity price shock — BOM weight × £3.3B commodity basket × shock %
              {isLive && <span className="ml-2 text-green-400">· Base EBIT from live pipeline</span>}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select value={shockCommodity} onChange={(e) => setShockCommodity(e.target.value)}
              className="rounded-lg px-3 py-1.5 border border-slate-600 text-slate-200 text-sm focus:outline-none focus:border-blue-500" style={{ backgroundColor: '#0f172a' }}>
              {SENSITIVITY.map((s) => <option key={s.commodity} value={s.commodity}>{s.commodity}</option>)}
            </select>
            <input type="range" min={-30} max={50} step={1} value={shockPct} onChange={(e) => setShockPct(Number(e.target.value))} className="w-40 accent-blue-500" />
            <span className={`font-bold text-sm w-12 text-right ${shockPct > 0 ? 'text-red-400' : shockPct < 0 ? 'text-green-400' : 'text-slate-300'}`}>
              {shockPct > 0 ? '+' : ''}{shockPct}%
            </span>
          </div>
        </div>

        <div className={`mb-4 rounded-lg px-4 py-3 border text-sm ${shockPct === 0 ? 'border-slate-700 text-slate-300' : shockPct > 0 ? 'border-red-800 bg-red-900/20 text-red-300' : 'border-green-800 bg-green-900/20 text-green-300'}`} style={shockPct === 0 ? { backgroundColor: '#0f172a' } : {}}>
          <span className="text-slate-400">{shockCommodity} {shockPct > 0 ? '+' : ''}{shockPct}%</span>
          <span className="mx-2 text-slate-600">→</span>
          EBIT Δ <span className="font-bold">{ebitDelta > 0 ? '+' : ''}£{ebitDelta.toLocaleString()}M</span>
          <span className="text-slate-500"> · EBIT £{shockedEBIT.toLocaleString()}M ({((shockedEBIT / baseEbit - 1) * 100).toFixed(1)}%)</span>
        </div>

        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={SENSITIVITY} layout="vertical" margin={{ top: 4, right: 20, left: 12, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} tickFormatter={(v) => `£${v}M`} />
            <YAxis type="category" dataKey="commodity" tick={{ fill: '#cbd5e1', fontSize: 12 }} axisLine={false} tickLine={false} width={84} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #475569', borderRadius: 8 }}
              formatter={(v, n, p) => [`£${(v).toFixed(1)}M per +1% · BOM ${p.payload.bomWeight}%`, 'EBIT impact']}
            />
            <ReferenceLine x={0} stroke="#64748b" />
            <Bar dataKey="impact1pct" radius={[0, 3, 3, 0]}>
              {SENSITIVITY.map((s, i) => <Cell key={i} fill={s.commodity === shockCommodity ? '#f59e0b' : '#ef4444'} fillOpacity={s.commodity === shockCommodity ? 1 : 0.55} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
