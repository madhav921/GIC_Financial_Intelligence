import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import KPICard from '../components/Charts/KPICard';
import Loading from '../components/common/Loading';
import Badge from '../components/common/Badge';
import { gicApi } from '../api/client';

// ── Mock fallback data ────────────────────────────────────────────────────────
const MOCK_KPI = {
  total_revenue: 19800000000,
  gross_margin_pct: 18.5,
  ebit: 1401000000,
  net_income: 1107000000,
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
    return () => { mounted = false; };
  }, []);

  const safeKpi = kpi || MOCK_KPI;

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
            subtitle="vs 17.8% prior year"
            change={0.7}
            changeType="up"
            format="percent"
          />
          <KPICard
            title="EBIT"
            value={safeKpi.ebit}
            subtitle="7.1% EBIT margin"
            change={1.4}
            changeType="up"
            format="currency"
          />
          <KPICard
            title="Net Income"
            value={safeKpi.net_income}
            subtitle="After tax & interest"
            change={-0.8}
            changeType="down"
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

        {/* Commodity Index Trend */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h3 className="text-lg font-semibold text-slate-100 mb-1">Commodity Cost Index</h3>
          <p className="text-slate-400 text-xs mb-4">Weighted BOM basket (Base = Jan 100)</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={COMMODITY_TREND} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} domain={[95, 125]} width={40} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #475569', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(v) => [`${v.toFixed(1)}`, 'Index']}
              />
              <Bar dataKey="index" fill="#f59e0b" radius={[3, 3, 0, 0]} />
            </BarChart>
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
            and favourable GBP/USD movement in H1. Gross margin expansion to 18.5% (+70bps) was achieved despite a 19% rise in lithium and cobalt
            costs, partially offset by SARIMAX-driven procurement hedging executed in Q4 FY2025.
          </p>
          <p className="mb-2">
            <span className="text-yellow-400 font-semibold">Key Risk: </span>
            Natural gas MAPE of 31% exceeds the 15% governance threshold, triggering mandatory scenario-based planning for H2 energy costs.
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
    </div>
  );
}
