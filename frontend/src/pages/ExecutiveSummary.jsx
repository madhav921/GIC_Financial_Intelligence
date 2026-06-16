import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import KPICard from '../components/Charts/KPICard';
import Loading from '../components/common/Loading';
import Badge from '../components/common/Badge';
import LiveMarketTape from '../components/realtime/LiveMarketTape';
import LiveKpiStrip from '../components/realtime/LiveKpiStrip';
import InsightCard from '../components/insights/InsightCard';
import RiskGauge from '../components/insights/RiskGauge';
import { useRealtimeContext } from '../context/RealtimeContext';
import { gicApi } from '../api/client';

// ── Mock fallback data ────────────────────────────────────────────────────────
// Ground-truth FY26 financials (single source of truth for all pages):
// Revenue £19.8B (+3.2% YoY) → Material COGS £12.7B → Gross Margin £7.1B (35.9%, −0.6pp YoY)
// → EBIT £1,401M (+8.3% YoY) → Net Finance Costs £180M → Pre-tax £1,221M → Tax 21% £256M
// → Net Income £965M (+9.8% YoY).  All pages must reference these constants.
const MOCK_KPI = {
  total_revenue: 19800000000,
  gross_margin_pct: 35.9,   // Gross Margin = (Revenue − Material COGS) / Revenue = 7100/19800
  ebit: 1401000000,
  net_income: 965000000,    // (EBIT − £180M finance costs) × (1 − 21% tax) = 1221 × 0.79
};

const MOCK_SEGMENTS = [
  { segment: 'Luxury SUV',  revenue: 8400000000, volume: 80000  },
  { segment: 'Premium SUV', revenue: 6640000000, volume: 120000 },
  { segment: 'Performance', revenue: 2960000000, volume: 65000  },
  { segment: 'EV',          revenue: 1800000000, volume: 45000  },
];

const MOCK_ALERTS = [
  {
    commodity: 'Lithium',
    type: 'Variance Alert',
    message: '+12.3% vs forecast',
    severity: 'warning',
  },
  {
    commodity: 'Natural Gas',
    type: 'High Volatility',
    message: 'MAPE 31% — use scenarios',
    severity: 'danger',
  },
  {
    commodity: 'Palladium',
    type: 'Regime Change',
    message: 'Trending detected (H=0.62)',
    severity: 'info',
  },
];

const COMMODITY_TREND = [
  { month: 'Jan', index: 100 }, { month: 'Feb', index: 103 },
  { month: 'Mar', index: 101 }, { month: 'Apr', index: 107 },
  { month: 'May', index: 109 }, { month: 'Jun', index: 106 },
  { month: 'Jul', index: 112 }, { month: 'Aug', index: 115 },
  { month: 'Sep', index: 113 }, { month: 'Oct', index: 118 },
  { month: 'Nov', index: 121 }, { month: 'Dec', index: 119 },
];

// ── Mock fallbacks for actionable-intelligence sections ───────────────────────
const MOCK_INSIGHTS = {
  insights: [
    {
      id: 'EX-1', category: 'Commodity', severity: 'critical', priority: 1,
      title: 'Lithium spike threatens EV battery margin',
      finding: 'Lithium is +12.3% vs plan, lifting EV pack cost by an estimated £74M against budget.',
      reasoning: 'Sustained spot breakout above the plan anchor; elasticity maps 1% input move to ~£6M EV BOM impact.',
      impact_gbp: -74000000, impact_label: '-£74.0M', confidence: 0.82,
      recommended_action: 'Execute the pre-approved 6-month lithium hedge and re-open supplier index clauses.',
      expected_action_savings_gbp: 41000000, affected_segments: ['EV', 'Performance'],
      supporting_metrics: { variance_pct: 12.3, spot_price: 15950 },
    },
    {
      id: 'EX-2', category: 'Warranty', severity: 'critical', priority: 1,
      title: 'EV battery failures running ahead of accrual',
      finding: 'Projected 12M warranty cost exceeds the booked accrual by £18M on 2024-build EV packs.',
      reasoning: 'Weibull hazard fit shows accelerating early-life failures; 7.4% shortfall vs modelled liability.',
      impact_gbp: -18000000, impact_label: '-£18.0M', confidence: 0.71,
      recommended_action: 'Top up the warranty accrual by £18M and launch an 8D on the cell supplier batch.',
      expected_action_savings_gbp: 12000000, affected_segments: ['EV'],
      supporting_metrics: { shortfall_pct: 7.4 },
    },
    {
      id: 'EX-3', category: 'Commodity', severity: 'warning', priority: 2,
      title: 'Aluminium softening — procurement timing opportunity',
      finding: 'Aluminium is forecast to ease 4-6%, opening a £19M cost-down on body structures.',
      reasoning: 'Improved supply and inventory builds drive the downward nowcast at 79% confidence.',
      impact_gbp: 19000000, impact_label: '+£19.0M', confidence: 0.79,
      recommended_action: 'Defer non-critical aluminium POs by 4-6 weeks to capture the dip.',
      expected_action_savings_gbp: 13000000, affected_segments: ['Luxury SUV', 'Premium SUV'],
      supporting_metrics: { forecast_change_pct: -5.0 },
    },
  ],
  summary: { n_critical: 2, n_warning: 3, total_impact_gbp: -80000000, total_opportunity_gbp: 59000000, weighted_confidence: 0.74 },
};

// Components now carry both score (0-100) and weight so the CFO can see
// WHAT is risky (score) AND how much it drives the composite (weight × score).
// Verify: 75×0.42 + 42×0.21 + 60×0.24 + 28×0.13 = 31.5+8.82+14.4+3.64 = 58.36 ≈ 58 ✓
const MOCK_EARLY_WARNING = {
  score: 58, band: 'elevated',
  components: {
    commodity: { score: 75, weight: 0.42 },
    warranty:  { score: 60, weight: 0.24 },
    margin:    { score: 42, weight: 0.21 },
    demand:    { score: 28, weight: 0.13 },
  },
  top_drivers: ['Lithium spike', 'EV warranty trend', 'EU demand softening'],
};

function fmtGBPex(v) {
  if (v === null || v === undefined) return '—';
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}£${(abs / 1e9).toFixed(2)}bn`;
  if (abs >= 1e6) return `${sign}£${(abs / 1e6).toFixed(1)}M`;
  return `${sign}£${abs.toFixed(0)}`;
}

const SEGMENT_COLORS = ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b'];

const ALERT_SEVERITY_STYLES = {
  warning: { bg: 'rgba(245,158,11,0.1)',  border: '#d97706', badge: 'yellow', icon: '⚠️' },
  danger:  { bg: 'rgba(239,68,68,0.1)',   border: '#dc2626', badge: 'red',    icon: '🔴' },
  info:    { bg: 'rgba(59,130,246,0.1)',  border: '#2563eb', badge: 'blue',   icon: 'ℹ️' },
};

const BarTooltipRevenue = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl" style={{ backgroundColor: '#0f172a' }}>
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      <p className="text-white">Revenue: <span className="font-mono font-bold">£{(payload[0].value / 1e9).toFixed(2)}B</span></p>
    </div>
  );
};

export default function ExecutiveSummary() {
  const [kpi, setKpi] = useState(null);
  const [segments, setSegments] = useState(MOCK_SEGMENTS);
  const [loading, setLoading] = useState(true);
  const [insightsData, setInsightsData] = useState(MOCK_INSIGHTS);
  const [earlyWarning, setEarlyWarning] = useState(MOCK_EARLY_WARNING);
  const { snapshot } = useRealtimeContext();

  useEffect(() => {
    let mounted = true;
    const fetchData = async () => {
      try {
        const data = await gicApi.getAnnualPnL();
        if (mounted) {
          setKpi(data);
          if (data.segments) setSegments(data.segments);
        }
      } catch {
        if (mounted) setKpi(MOCK_KPI);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    fetchData();
    // Actionable-intelligence sections (independent; degrade to mock)
    (async () => {
      try {
        const feed = await gicApi.insightsFeed();
        if (mounted && feed?.insights) setInsightsData(feed);
      } catch { /* keep mock */ }
    })();
    (async () => {
      try {
        const ew = await gicApi.earlyWarning();
        if (mounted && ew?.score !== undefined) setEarlyWarning(ew);
      } catch { /* keep mock */ }
    })();
    return () => { mounted = false; };
  }, []);

  const safeKpi = kpi || MOCK_KPI;
  const feed = insightsData || MOCK_INSIGHTS;
  const insightSummary = feed.summary || MOCK_INSIGHTS.summary;
  const topInsights = (feed.insights || []).slice(0, 3);
  const ew = earlyWarning || MOCK_EARLY_WARNING;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className="text-blue-400">ℹ️</span>
        Connect backend:{' '}
        <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code>
        {' '}— showing mock data while offline.
      </div>

      {/* Page title */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Executive Summary</h2>
        <p className="text-slate-400 text-sm">GIC Plan-to-Perform · FY 2026 Consolidated View</p>
      </div>

      {/* Live market tape (full width) */}
      <LiveMarketTape />

      {/* Live ticking KPIs */}
      <LiveKpiStrip />

      {/* KPI Cards */}
      {loading ? (
        <Loading message="Loading financial data..." />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Total Revenue"
            value={safeKpi.total_revenue}
            subtitle="FY 2026 Consolidated"
            change={3.2}
            changeType="up"
            format="currency"
          />
          <KPICard
            title="Gross Margin"
            value={safeKpi.gross_margin_pct}
            subtitle="vs 36.5% prior year (Revenue − Material COGS)"
            change={-0.6}
            changeType="down"
            format="percent"
          />
          <KPICard
            title="EBIT"
            value={safeKpi.ebit}
            subtitle="7.1% EBIT margin"
            change={8.3}
            changeType="up"
            format="currency"
          />
          <KPICard
            title="Net Income"
            value={safeKpi.net_income}
            subtitle="After £180M finance costs & 21% tax"
            change={9.8}
            changeType="up"
            format="currency"
          />
        </div>
      )}

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Segment Revenue Bar Chart */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h3 className="text-lg font-semibold text-slate-100 mb-4">Revenue by Segment</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={segments} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="segment" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `£${(v / 1e9).toFixed(0)}B`} width={50} />
              <Tooltip content={<BarTooltipRevenue />} />
              <Bar dataKey="revenue" radius={[4, 4, 0, 0]}>
                {segments.map((_, i) => (
                  <Cell key={i} fill={SEGMENT_COLORS[i % SEGMENT_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Commodity Index Trend — line chart (time-series, not categorical) */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h3 className="text-lg font-semibold text-slate-100 mb-1">Commodity Cost Index</h3>
          <p className="text-slate-400 text-xs mb-4">Weighted BOM basket · Base = Jan 100 · +19% YTD</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={COMMODITY_TREND} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} domain={[95, 125]} width={40} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #475569', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(v) => [`${v.toFixed(1)}`, 'Index']}
              />
              <Line type="monotone" dataKey="index" stroke="#f59e0b" strokeWidth={2.5} dot={{ r: 3, fill: '#f59e0b' }} activeDot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Alerts + Segment Table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Alerts */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h3 className="text-lg font-semibold text-slate-100 mb-4">Active Risk Alerts</h3>
          <div className="space-y-3">
            {MOCK_ALERTS.map((alert, i) => {
              const style = ALERT_SEVERITY_STYLES[alert.severity] || ALERT_SEVERITY_STYLES.info;
              return (
                <div key={i} className="rounded-lg px-4 py-3 border flex items-start gap-3" style={{ backgroundColor: style.bg, borderColor: style.border }}>
                  <span className="text-base mt-0.5 flex-shrink-0">{style.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-semibold text-slate-100">{alert.commodity}</span>
                      <Badge label={alert.type} color={style.badge} />
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">{alert.message}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Segment Table */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h3 className="text-lg font-semibold text-slate-100 mb-4">Segment Performance</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="text-left text-xs text-slate-400 font-medium pb-2">Segment</th>
                <th className="text-right text-xs text-slate-400 font-medium pb-2">Revenue</th>
                <th className="text-right text-xs text-slate-400 font-medium pb-2">Volume</th>
                <th className="text-right text-xs text-slate-400 font-medium pb-2">Mix %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {segments.map((seg, i) => {
                const totalRev = segments.reduce((s, x) => s + (x.revenue || 0), 0);
                const mix = totalRev > 0 ? ((seg.revenue / totalRev) * 100).toFixed(1) : '0.0';
                return (
                  <tr key={i} className="hover:bg-slate-700/30 transition-colors">
                    <td className="py-2.5 text-slate-200">
                      <span className="inline-block w-2 h-2 rounded-full mr-2" style={{ backgroundColor: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }} />
                      {seg.segment}
                    </td>
                    <td className="py-2.5 text-right font-mono text-slate-300">
                      £{(seg.revenue / 1e9).toFixed(2)}B
                    </td>
                    <td className="py-2.5 text-right font-mono text-slate-300">
                      {(seg.volume / 1000).toFixed(0)}K
                    </td>
                    <td className="py-2.5 text-right font-mono text-slate-400">{mix}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* AI Insight */}
      <div className="rounded-xl p-6 border border-blue-900/50" style={{ backgroundColor: '#1e293b' }}>
        <div className="flex items-center gap-3 mb-4">
          <span className="text-xl">🤖</span>
          <h3 className="text-lg font-semibold text-slate-100">AI Narrative — Executive Brief</h3>
          <Badge label="LLM Generated" color="blue" />
        </div>
        <div className="rounded-lg p-4 border border-slate-700 text-sm text-slate-300 leading-relaxed" style={{ backgroundColor: '#0f172a' }}>
          <p className="mb-2">
            <span className="text-blue-400 font-semibold">Performance Overview: </span>
            FY 2026 consolidated revenue of £19.8B reflects a 3.2% year-on-year improvement, driven primarily by Luxury SUV segment volume recovery
            and favourable GBP/USD movement in H1. Gross margin (Revenue − Material COGS) contracted to 35.9% (−60bps vs prior year 36.5%) as a 19% rise in lithium and cobalt
            costs was only partially offset by SARIMAX-driven procurement hedging executed in Q4 FY2025. EBIT grew +8.3% to £1,401M as operating leverage absorbed the commodity headwind.
          </p>
          <p className="mb-2">
            <span className="text-yellow-400 font-semibold">Key Risk: </span>
            Natural gas MAPE of 31% exceeds the 20% high-volatility governance threshold, triggering mandatory scenario-based planning for H2 energy costs.
            The commodity cost index reached 119 in December (+19% YTD), with lithium variance of +12.3% vs forecast representing the largest
            single BOM exposure.
          </p>
          <p>
            <span className="text-green-400 font-semibold">Outlook: </span>
            The Palladium regime-change signal (Hurst exponent 0.62) suggests a trending environment warranting model weight rebalancing toward
            XGBoost. EV segment revenue of £1.8B is tracking ahead of plan by 8%, supported by government subsidy tailwinds in key European markets.
          </p>
        </div>
        <p className="text-xs text-slate-500 mt-3">
          Generated by google/flan-t5-base · Model confidence: High · Last run: 11 Jun 2026 09:05
        </p>
      </div>

      {/* ── Top Actionable Insights ─────────────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-100">Top Actionable Insights</h3>
            <p className="text-slate-400 text-xs mt-0.5">Highest-impact, prescriptive recommendations</p>
          </div>
          <div className="flex items-center gap-2 flex-wrap text-xs">
            <Badge label={`${insightSummary.n_critical} Critical`} color="red" />
            <Badge label={`${insightSummary.n_warning} Warning`} color="yellow" />
            <span className="px-2 py-0.5 rounded-full border border-slate-600 text-slate-300">
              Net impact <span className="font-mono font-semibold">{fmtGBPex(insightSummary.total_impact_gbp)}</span>
            </span>
            <span className="px-2 py-0.5 rounded-full border border-emerald-700 text-emerald-300">
              Upside <span className="font-mono font-semibold">{fmtGBPex(insightSummary.total_opportunity_gbp)}</span>
            </span>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {topInsights.map((ins) => (
            <InsightCard key={ins.id} insight={ins} />
          ))}
        </div>
      </div>

      {/* ── Risk gauge + AI executive narrative ─────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="rounded-xl p-6 border border-slate-700 flex flex-col items-center justify-center" style={{ backgroundColor: '#1e293b' }}>
          <h3 className="text-lg font-semibold text-slate-100 mb-2 self-start">Early-Warning Risk</h3>
          <RiskGauge score={ew.score} band={ew.band} size={240} label="Composite Score" />
          {/* Component score breakdown: shows WHAT is risky (score) and its weight in composite */}
          {ew.components && typeof Object.values(ew.components)[0] === 'object' && (
            <div className="mt-4 w-full">
              <div className="text-[11px] text-slate-500 uppercase tracking-wide mb-2">Risk components (score × weight)</div>
              <div className="space-y-1.5">
                {Object.entries(ew.components).map(([key, val]) => {
                  const score = val?.score ?? val;
                  const weight = val?.weight ?? null;
                  const contribution = weight != null ? (score * weight).toFixed(1) : null;
                  const barColor = score >= 70 ? '#ef4444' : score >= 45 ? '#f59e0b' : '#22c55e';
                  return (
                    <div key={key}>
                      <div className="flex justify-between text-[11px] mb-0.5">
                        <span className="text-slate-400 capitalize">{key}</span>
                        <span className="text-slate-300 font-mono">
                          {score}/100{weight != null ? ` · ${(weight * 100).toFixed(0)}% wt` : ''}{contribution != null ? ` = ${contribution} pts` : ''}
                        </span>
                      </div>
                      <div className="h-1 rounded-full bg-slate-700">
                        <div className="h-1 rounded-full" style={{ width: `${score}%`, backgroundColor: barColor }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          {ew.top_drivers?.length > 0 && (
            <div className="mt-3 w-full">
              <div className="text-[11px] text-slate-500 uppercase tracking-wide mb-1.5">Top drivers</div>
              <div className="flex flex-wrap gap-1.5">
                {ew.top_drivers.slice(0, 3).map((dr, i) => {
                  const label = typeof dr === 'string' ? dr : (dr?.component || String(dr));
                  return (
                    <span key={i} className="text-[11px] px-2 py-0.5 rounded-full bg-slate-700/60 text-slate-300 border border-slate-600">{label}</span>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-2 rounded-xl p-6 border border-blue-900/50" style={{ backgroundColor: '#1e293b' }}>
          <div className="flex items-center gap-3 mb-3">
            <span className="text-xl">🤖</span>
            <h3 className="text-lg font-semibold text-slate-100">AI Executive Insight</h3>
            <Badge label="Live synthesis" color="blue" />
          </div>
          <div className="rounded-lg p-4 border border-slate-700 text-sm text-slate-300 leading-relaxed" style={{ backgroundColor: '#0f172a' }}>
            <p className="mb-2">
              <span className="text-blue-400 font-semibold">Headline: </span>
              {snapshot?.headline_insight || 'Commodity index stable; risk band holding within tolerance.'}
            </p>
            <p>
              The early-warning model places composite risk at{' '}
              <span className="font-semibold text-white">{Number(ew.score).toFixed(0)}/100</span>{' '}
              (<span className="capitalize">{ew.band}</span>), driven principally by{' '}
              <span className="text-amber-300 font-semibold">{
                (() => { const d = ew.top_drivers?.[0]; return (typeof d === 'string' ? d : d?.component) || 'commodity exposure'; })()
              }</span>.
              Against a net exposure of <span className="font-mono text-red-300">{fmtGBPex(insightSummary.total_impact_gbp)}</span>,
              the insight engine identifies <span className="font-mono text-emerald-300">{fmtGBPex(insightSummary.total_opportunity_gbp)}</span> of
              addressable upside — concentrated in hedging the lithium spike and timing aluminium procurement. EBIT nowcast currently reads{' '}
              <span className="font-mono text-blue-300">£{((snapshot?.ebit_nowcast_gbp ?? 1.4e9) / 1e9).toFixed(2)}bn</span>.
            </p>
          </div>
          <p className="text-xs text-slate-500 mt-3">
            Synthesised from realtime snapshot, insights feed and early-warning model · {new Date().toLocaleString('en-GB')}
          </p>
        </div>
      </div>
    </div>
  );
}
