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

const COMMODITY_TABLE = [
  { name: 'Steel',         current: '789.7',     unit: 'USD/t',    bomWt: '22%', annSpend: '$1,870M', chg1m: '+1.8%',  chg3m: '+4.2%',   chg12m: '+72.9%',  vsAvg: '+97.7%',  vol: '31%', regime: 'Strong Uptrend',    val: 'Historically Expensive' },
  { name: 'Lithium',       current: '20.8',      unit: 'USD/kg',   bomWt: '18%', annSpend: '$1,530M', chg1m: '−5.7%',  chg3m: '+10.4%',  chg12m: '+131.7%', vsAvg: '+57.6%',  vol: '32%', regime: 'Strong Uptrend',    val: 'Historically Expensive' },
  { name: 'Aluminum',      current: '3,735.0',   unit: 'USD/t',    bomWt: '12%', annSpend: '$1,020M', chg1m: '−2.4%',  chg3m: '+0.4%',   chg12m: '+135.1%', vsAvg: '+78.0%',  vol: '62%', regime: 'Strong Uptrend',    val: 'Historically Expensive' },
  { name: 'Cobalt',        current: '24,688.0',  unit: 'USD/t',    bomWt: '7%',  annSpend: '$595M',   chg1m: '+1.1%',  chg3m: '+8.8%',   chg12m: '+108.2%', vsAvg: '+85.9%',  vol: '39%', regime: 'Strong Uptrend',    val: 'Historically Expensive' },
  { name: 'Copper',        current: '13,904.5',  unit: 'USD/t',    bomWt: '6%',  annSpend: '$510M',   chg1m: '+6.4%',  chg3m: '+7.0%',   chg12m: '+35.6%',  vsAvg: '+58.5%',  vol: '20%', regime: 'Strong Uptrend',    val: 'Historically Expensive' },
  { name: 'Nickel',        current: '12,236.5',  unit: 'USD/t',    bomWt: '5%',  annSpend: '$425M',   chg1m: '−0.3%',  chg3m: '−5.0%',   chg12m: '+94.7%',  vsAvg: '+65.0%',  vol: '38%', regime: 'Strong Uptrend',    val: 'Historically Expensive' },
  { name: 'Platinum',      current: '1,971.8',   unit: 'USD/oz',   bomWt: '4%',  annSpend: '$340M',   chg1m: '−0.4%',  chg3m: '−6.2%',   chg12m: '+87.5%',  vsAvg: '+83.5%',  vol: '25%', regime: 'Uptrend',           val: 'Historically Expensive' },
  { name: 'Natural Gas',   current: '30.1',      unit: 'p/therm',  bomWt: '4%',  annSpend: '$340M',   chg1m: '+8.9%',  chg3m: '−30.8%',  chg12m: '−12.6%', vsAvg: '−12.5%',  vol: '63%', regime: 'Downtrend',         val: 'Near Fair Value' },
  { name: 'Polypropylene', current: '938.5',     unit: 'USD/t',    bomWt: '3%',  annSpend: '$255M',   chg1m: '+0.7%',  chg3m: '−15.2%',  chg12m: '+1.0%',   vsAvg: '−12.8%',  vol: '26%', regime: 'Strong Downtrend',  val: 'Below Average' },
  { name: 'Palladium',     current: '1,419.5',   unit: 'USD/oz',   bomWt: '3%',  annSpend: '$255M',   chg1m: '−7.0%',  chg3m: '−15.9%',  chg12m: '+47.7%',  vsAvg: '−15.8%',  vol: '35%', regime: 'Downtrend',         val: 'Near Fair Value' },
  { name: 'Rhodium',       current: '3,936.6',   unit: 'USD/oz',   bomWt: '2%',  annSpend: '$170M',   chg1m: '−2.6%',  chg3m: '−5.0%',   chg12m: '−14.1%', vsAvg: '−13.7%',  vol: '27%', regime: 'Downtrend',         val: 'Below Average' },
  { name: 'ABS Resin',     current: '1,438.1',   unit: 'USD/t',    bomWt: '2%',  annSpend: '$170M',   chg1m: '+16.7%', chg3m: '+0.9%',   chg12m: '−10.1%', vsAvg: '−19.7%',  vol: '32%', regime: 'Downtrend',         val: 'Below Average' },
];

const SCENARIO_TABLE = [
  { name: 'Steel',         bomWt: '22%', current: '$790',     bear: '$380 (−52%)',  base: '$510 (−35%)',  bull: '$640 (−19%)',  bearDelta: '−$970M',   baseDelta: '−$662M', bullDelta: '−$354M' },
  { name: 'Lithium',       bomWt: '18%', current: '$21/kg',   bear: '$7 (−66%)',    base: '$12 (−42%)',   bull: '$20 (−4%)',    bearDelta: '−$1,015M', baseDelta: '−$648M', bullDelta: '−$60M' },
  { name: 'Aluminum',      bomWt: '12%', current: '$3,735',   bear: '$1,900 (−49%)',base: '$2,500 (−33%)',bull: '$3,100 (−17%)',bearDelta: '−$501M',   baseDelta: '−$337M', bullDelta: '−$173M' },
  { name: 'Cobalt',        bomWt: '7%',  current: '$24,688',  bear: '$18,000 (−27%)',base: '$28,000 (+13%)',bull: '$40,000 (+62%)',bearDelta: '−$161M', baseDelta: '+$80M',  bullDelta: '+$369M' },
  { name: 'Copper',        bomWt: '6%',  current: '$13,905',  bear: '$8,200 (−41%)',base: '$10,500 (−24%)',bull: '$12,800 (−8%)', bearDelta: '−$209M', baseDelta: '−$125M', bullDelta: '−$40M' },
  { name: 'Nickel',        bomWt: '5%',  current: '$12,236',  bear: '$13,000 (+6%)',base: '$18,000 (+47%)',bull: '$24,000 (+96%)',bearDelta: '+$26M',  baseDelta: '+$200M', bullDelta: '+$409M' },
  { name: 'Platinum',      bomWt: '4%',  current: '$1,972',   bear: '$700 (−65%)',  base: '$1,050 (−47%)',bull: '$1,400 (−29%)', bearDelta: '−$219M', baseDelta: '−$159M', bullDelta: '−$99M' },
  { name: 'Natural Gas',   bomWt: '4%',  current: '30p/th',   bear: '20p (−34%)',   base: '38p (+26%)',   bull: '60p (+99%)',   bearDelta: '−$114M',   baseDelta: '+$89M',  bullDelta: '+$337M' },
  { name: 'Polypropylene', bomWt: '3%',  current: '$938',     bear: '$900 (−4%)',   base: '$1,300 (+38%)',bull: '$1,600 (+70%)', bearDelta: '−$10M',  baseDelta: '+$98M',  bullDelta: '+$180M' },
  { name: 'Palladium',     bomWt: '3%',  current: '$1,420',   bear: '$700 (−51%)',  base: '$1,100 (−22%)',bull: '$1,500 (+6%)', bearDelta: '−$129M',   baseDelta: '−$57M',  bullDelta: '+$14M' },
  { name: 'Rhodium',       bomWt: '2%',  current: '$3,937',   bear: '$3,000 (−24%)',base: '$5,000 (+27%)',bull: '$7,500 (+90%)', bearDelta: '−$40M',  baseDelta: '+$46M',  bullDelta: '+$154M' },
  { name: 'ABS Resin',     bomWt: '2%',  current: '$1,438',   bear: '$1,100 (−24%)',base: '$1,600 (+11%)',bull: '$2,000 (+39%)', bearDelta: '−$40M',  baseDelta: '+$19M',  bullDelta: '+$66M' },
];

function buildReportHTML({ kpi, segments, ew, insightSummary, topInsights }) {
  const d = new Date();
  const reportDate = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
  const reportTime = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  const rev = kpi?.total_revenue ?? 19800000000;
  const ebit = kpi?.ebit ?? 1401000000;
  const gm = kpi?.gross_margin_pct ?? 35.9;
  const ni = kpi?.net_income ?? 965000000;
  const fmtGBP = (v) => { const a = Math.abs(v); const s = v < 0 ? '−' : ''; return a >= 1e9 ? `${s}£${(a/1e9).toFixed(2)}bn` : `${s}£${(a/1e6).toFixed(0)}M`; };
  const riskColor = ew?.score >= 70 ? '#dc2626' : ew?.score >= 45 ? '#d97706' : '#16a34a';
  const riskBg   = ew?.score >= 70 ? '#fef2f2' : ew?.score >= 45 ? '#fffbeb' : '#f0fdf4';

  const commodityRows = COMMODITY_TABLE.map(c => {
    const up12 = c.chg12m.startsWith('+');
    const dn12 = c.chg12m.startsWith('−') || c.chg12m.startsWith('-');
    const chgColor = up12 ? '#dc2626' : dn12 ? '#16a34a' : '#374151';
    const valColor = c.val === 'Historically Expensive' ? '#dc2626' : c.val === 'Below Average' ? '#16a34a' : '#374151';
    return `<tr>
      <td style="font-weight:600;padding:7px 10px;border-bottom:1px solid #e5e7eb">${c.name}</td>
      <td style="font-family:monospace;padding:7px 10px;border-bottom:1px solid #e5e7eb">${c.current} <span style="color:#6b7280;font-size:11px">${c.unit}</span></td>
      <td style="text-align:center;padding:7px 10px;border-bottom:1px solid #e5e7eb">${c.bomWt}</td>
      <td style="font-family:monospace;padding:7px 10px;border-bottom:1px solid #e5e7eb">${c.annSpend}</td>
      <td style="text-align:center;padding:7px 10px;border-bottom:1px solid #e5e7eb;color:${chgColor};font-weight:600">${c.chg12m}</td>
      <td style="text-align:center;padding:7px 10px;border-bottom:1px solid #e5e7eb">${c.vsAvg}</td>
      <td style="text-align:center;padding:7px 10px;border-bottom:1px solid #e5e7eb">${c.vol}</td>
      <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;color:${valColor};font-size:11px">${c.val}</td>
    </tr>`;
  }).join('');

  const scenarioRows = SCENARIO_TABLE.map(s => {
    const bearIsNeg = s.bearDelta.startsWith('−');
    return `<tr>
      <td style="font-weight:600;padding:7px 10px;border-bottom:1px solid #e5e7eb">${s.name}</td>
      <td style="text-align:center;padding:7px 10px;border-bottom:1px solid #e5e7eb">${s.bomWt}</td>
      <td style="font-family:monospace;padding:7px 10px;border-bottom:1px solid #e5e7eb">${s.current}</td>
      <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;color:#374151">${s.bear}</td>
      <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;color:#374151">${s.base}</td>
      <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;color:#374151">${s.bull}</td>
      <td style="font-family:monospace;font-weight:600;padding:7px 10px;border-bottom:1px solid #e5e7eb;color:${bearIsNeg?'#dc2626':'#16a34a'}">${s.bearDelta}</td>
      <td style="font-family:monospace;padding:7px 10px;border-bottom:1px solid #e5e7eb">${s.baseDelta}</td>
    </tr>`;
  }).join('');

  const insightCards = (topInsights || []).map(ins => {
    const isCrit = ins.severity === 'critical';
    const impact = ins.impact_gbp ?? 0;
    return `<div style="border:1px solid ${isCrit?'#fca5a5':'#fde68a'};border-radius:8px;padding:14px;background:${isCrit?'#fff5f5':'#fffbeb'}">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="font-size:11px;font-weight:700;padding:2px 8px;border-radius:12px;background:${isCrit?'#fee2e2':'#fef3c7'};color:${isCrit?'#b91c1c':'#92400e'}">${isCrit?'CRITICAL':'WARNING'}</span>
        <span style="font-size:11px;color:#6b7280">${ins.category}</span>
      </div>
      <p style="font-weight:600;font-size:13px;margin:0 0 4px">${ins.title}</p>
      <p style="font-size:12px;color:#374151;margin:0 0 8px">${ins.finding}</p>
      <p style="font-size:12px;margin:0"><strong>Action:</strong> ${ins.recommended_action}</p>
      <p style="font-size:12px;color:${impact<0?'#dc2626':'#16a34a'};font-family:monospace;margin:4px 0 0;font-weight:600">
        Impact: ${fmtGBP(impact)}${ins.expected_action_savings_gbp ? ` · Savings if actioned: ${fmtGBP(ins.expected_action_savings_gbp)}` : ''}
      </p>
    </div>`;
  }).join('');

  const segRows = (segments || []).map((seg, i) => {
    const totalRev = (segments || []).reduce((s, x) => s + (x.revenue || 0), 0);
    const mix = totalRev > 0 ? ((seg.revenue / totalRev) * 100).toFixed(1) : '0.0';
    const colors = ['#3b82f6','#8b5cf6','#22c55e','#f59e0b'];
    return `<tr>
      <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb">
        <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${colors[i%4]};margin-right:8px"></span>
        ${seg.segment}
      </td>
      <td style="font-family:monospace;text-align:right;padding:7px 10px;border-bottom:1px solid #e5e7eb">£${(seg.revenue/1e9).toFixed(2)}B</td>
      <td style="font-family:monospace;text-align:right;padding:7px 10px;border-bottom:1px solid #e5e7eb">${((seg.volume||0)/1000).toFixed(0)}K</td>
      <td style="text-align:right;padding:7px 10px;border-bottom:1px solid #e5e7eb;color:#6b7280">${mix}%</td>
    </tr>`;
  }).join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GIC Executive Intelligence Report — ${reportDate}</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;color:#111827;background:#f9fafb;line-height:1.5}
  .page{max-width:1100px;margin:0 auto;background:#fff;padding:48px;min-height:100vh}
  h1{font-size:22px;font-weight:800;color:#111827;margin-bottom:4px}
  h2{font-size:16px;font-weight:700;color:#1e3a5f;margin:32px 0 14px;padding-bottom:6px;border-bottom:2px solid #1e3a5f}
  h3{font-size:13px;font-weight:700;color:#374151;margin:18px 0 10px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{background:#1e3a5f;color:#fff;padding:8px 10px;text-align:left;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
  th.r,td.r{text-align:right}
  th.c,td.c{text-align:center}
  .kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
  .kpi{border:1px solid #e5e7eb;border-radius:10px;padding:16px;background:#fff}
  .kpi-label{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px}
  .kpi-value{font-size:22px;font-weight:800;font-family:monospace;color:#111827}
  .kpi-sub{font-size:11px;color:#6b7280;margin-top:2px}
  .kpi-change-up{color:#16a34a;font-size:11px;font-weight:600}
  .kpi-change-down{color:#dc2626;font-size:11px;font-weight:600}
  .alert-box{border-left:4px solid;padding:12px 16px;border-radius:0 8px 8px 0;margin-bottom:10px}
  .risk-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700}
  .section-intro{color:#4b5563;font-size:12px;margin-bottom:16px;padding:12px;background:#f8fafc;border-left:3px solid #3b82f6;border-radius:0 6px 6px 0}
  .footer{border-top:1px solid #e5e7eb;margin-top:48px;padding-top:16px;font-size:11px;color:#9ca3af;display:flex;justify-content:space-between}
  .header-meta{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:32px}
  .header-meta-right{text-align:right;font-size:11px;color:#6b7280}
  .confidential{background:#fef2f2;color:#b91c1c;border:1px solid #fca5a5;padding:4px 12px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:1px}
  .action-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
  .action-card{border:1px solid #e5e7eb;border-radius:8px;padding:14px}
  .insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  .scenario-summary-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}
  .scenario-card{border-radius:10px;padding:16px;text-align:center}
  @media print{body{background:#fff}.page{padding:24px;max-width:100%}}
  @page{margin:20mm;size:A4 landscape}
  tr:hover{background:#f9fafb}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <div class="header-meta">
    <div>
      <h1>GIC Plan-to-Perform</h1>
      <p style="color:#1e3a5f;font-size:16px;font-weight:600;margin-top:2px">Board Intelligence Brief — Commodity Risk & Financial Impact</p>
      <p style="color:#6b7280;font-size:12px;margin-top:6px">Data Range: June 2019 – May 2026 · 84 Monthly Observations · Sources: Yahoo Finance, FRED, O-U Model</p>
    </div>
    <div class="header-meta-right">
      <div class="confidential">CONFIDENTIAL</div>
      <p style="margin-top:8px">Generated: ${reportDate} at ${reportTime}</p>
      <p>GIC Engine v0.3.0</p>
      <p style="color:#1e3a5f;font-weight:600">Board / C-Suite / Supply Chain Leadership</p>
    </div>
  </div>

  <!-- KPIs -->
  <h2>Financial Performance — FY 2026</h2>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Total Revenue</div>
      <div class="kpi-value">£${(rev/1.27/1e9).toFixed(1)}B</div>
      <div class="kpi-sub">FY 2026 Consolidated</div>
      <div class="kpi-change-up">▲ +3.2% YoY</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Gross Margin</div>
      <div class="kpi-value">${typeof gm === 'number' ? gm.toFixed(1) : gm}%</div>
      <div class="kpi-sub">Revenue − Material COGS</div>
      <div class="kpi-change-down">▼ −0.6pp vs prior year</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">EBIT</div>
      <div class="kpi-value">£${(ebit/1.27/1e9).toFixed(2)}B</div>
      <div class="kpi-sub">7.1% EBIT margin</div>
      <div class="kpi-change-up">▲ +8.3% YoY</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Net Income</div>
      <div class="kpi-value">£${(ni/1.27/1e9).toFixed(2)}B</div>
      <div class="kpi-sub">After finance costs & tax</div>
      <div class="kpi-change-up">▲ +9.8% YoY</div>
    </div>
  </div>

  <!-- Risk headline -->
  <div style="display:grid;grid-template-columns:1fr 2fr;gap:16px;margin-bottom:24px">
    <div style="border:1px solid #e5e7eb;border-radius:10px;padding:20px;text-align:center;background:${riskBg}">
      <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#6b7280;margin-bottom:8px">Composite Risk Score</div>
      <div style="font-size:56px;font-weight:900;color:${riskColor};line-height:1">${ew?.score ?? 58}</div>
      <div style="font-size:13px;font-weight:700;color:${riskColor};text-transform:uppercase;margin-top:4px">${ew?.band ?? 'Elevated'}</div>
      <div style="font-size:11px;color:#6b7280;margin-top:8px">0 = Safe · 100 = Critical</div>
    </div>
    <div style="border:1px solid #e5e7eb;border-radius:10px;padding:20px">
      <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#6b7280;margin-bottom:10px">Commodity Cost Index — May 2026</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
        <div style="background:#fef2f2;border-radius:8px;padding:10px"><div style="font-size:11px;color:#6b7280">Current Index</div><div style="font-size:24px;font-weight:800;color:#dc2626">272.4</div><div style="font-size:11px;color:#dc2626">+84.7% YoY</div></div>
        <div style="background:#f0fdf4;border-radius:8px;padding:10px"><div style="font-size:11px;color:#6b7280">Base-Case COGS Δ (12m)</div><div style="font-size:24px;font-weight:800;color:#dc2626">−$1,456M</div><div style="font-size:11px;color:#dc2626">−6.1% EBIT impact</div></div>
        <div style="background:#fff7ed;border-radius:8px;padding:10px"><div style="font-size:11px;color:#6b7280">Annual Material Spend</div><div style="font-size:20px;font-weight:800;color:#92400e">$8.5B</div><div style="font-size:11px;color:#6b7280">45% of COGS</div></div>
        <div style="background:#fef2f2;border-radius:8px;padding:10px"><div style="font-size:11px;color:#6b7280">Bear-Case COGS Δ (12m)</div><div style="font-size:20px;font-weight:800;color:#dc2626">−$3,384M</div><div style="font-size:11px;color:#dc2626">−14.1% EBIT impact</div></div>
      </div>
    </div>
  </div>

  <!-- Actionable Insights -->
  ${topInsights?.length > 0 ? `
  <h2>Top Actionable Insights</h2>
  <p class="section-intro">
    ${insightSummary?.n_critical ?? 0} critical · ${insightSummary?.n_warning ?? 0} warning ·
    Net impact: <strong style="color:#dc2626">${fmtGBP(insightSummary?.total_impact_gbp ?? -80000000)}</strong> ·
    Addressable upside: <strong style="color:#16a34a">${fmtGBP(insightSummary?.total_opportunity_gbp ?? 59000000)}</strong>
  </p>
  <div class="insight-grid">${insightCards}</div>
  ` : ''}

  <!-- Segment Revenue -->
  <h2>Segment Performance</h2>
  <table>
    <thead><tr><th>Segment</th><th class="r">Revenue</th><th class="r">Volume</th><th class="r">Revenue Mix</th></tr></thead>
    <tbody>${segRows}</tbody>
  </table>

  <!-- Commodity Dashboard -->
  <h2>Commodity Price Dashboard</h2>
  <p class="section-intro">All prices sourced from live market feeds. Rhodium, Polypropylene, ABS Resin: modelled via mean-reverting O-U process (calibrated to LPPM/ICIS indices).</p>
  <table>
    <thead>
      <tr>
        <th>Commodity</th><th>Price</th><th class="c">BOM Wt</th><th class="c">Ann. Spend</th>
        <th class="c">12M Δ</th><th class="c">vs 7yr Avg</th><th class="c">Ann. Vol</th><th>Valuation</th>
      </tr>
    </thead>
    <tbody>${commodityRows}</tbody>
  </table>

  <!-- Scenario Analysis -->
  <h2>Scenario Analysis — 12-Month P&L Impact</h2>
  <div class="scenario-summary-grid">
    <div class="scenario-card" style="background:#fef2f2;border:1px solid #fca5a5">
      <div style="font-size:24px;margin-bottom:4px">🐻</div>
      <div style="font-weight:700;font-size:13px">Bear (Contraction)</div>
      <div style="font-size:11px;color:#6b7280;margin:4px 0">PMI &lt; 48 · China GDP 4.0% · Energy spike</div>
      <div style="font-size:28px;font-weight:900;color:#dc2626;font-family:monospace">−$3,384M</div>
      <div style="font-size:12px;color:#dc2626;font-weight:600">−14.1% EBIT margin impact</div>
    </div>
    <div class="scenario-card" style="background:#fffbeb;border:1px solid #fde68a">
      <div style="font-size:24px;margin-bottom:4px">➡</div>
      <div style="font-weight:700;font-size:13px">Base (Consensus)</div>
      <div style="font-size:11px;color:#6b7280;margin:4px 0">PMI 51 · China GDP 4.75% · Energy stable</div>
      <div style="font-size:28px;font-weight:900;color:#d97706;font-family:monospace">−$1,456M</div>
      <div style="font-size:12px;color:#d97706;font-weight:600">−6.1% EBIT margin impact</div>
    </div>
    <div class="scenario-card" style="background:#f0fdf4;border:1px solid #86efac">
      <div style="font-size:24px;margin-bottom:4px">🐂</div>
      <div style="font-weight:700;font-size:13px">Bull (Expansion)</div>
      <div style="font-size:11px;color:#6b7280;margin:4px 0">PMI &gt; 53 · China GDP 5.5% · EV surge</div>
      <div style="font-size:28px;font-weight:900;color:#16a34a;font-family:monospace">+$803M</div>
      <div style="font-size:12px;color:#16a34a;font-weight:600">+3.3% EBIT margin impact</div>
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Commodity</th><th class="c">BOM Wt</th><th>Current</th>
        <th>Bear Target</th><th>Base Target</th><th>Bull Target</th>
        <th class="r">Bear COGS Δ</th><th class="r">Base COGS Δ</th>
      </tr>
    </thead>
    <tbody>${scenarioRows}</tbody>
  </table>

  <!-- Strategic Recommendations -->
  <h2>Strategic Recommendations</h2>
  <div class="action-grid">
    <div class="action-card" style="border-left:4px solid #dc2626">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-size:11px;font-weight:700;background:#fef2f2;color:#b91c1c;padding:2px 8px;border-radius:12px">P1 — IMMEDIATE</span>
        <span style="font-size:12px;font-weight:700">Steel</span>
      </div>
      <p style="font-size:12px;color:#374151"><strong>Signal:</strong> +72.9% YoY · +97.7% above 7yr mean · 31% vol</p>
      <p style="font-size:12px;color:#374151;margin-top:4px"><strong>Action:</strong> 12-month fixed-price contracts for up to 50% of $1,870M spend. BOM substitution study.</p>
      <p style="font-size:11px;color:#dc2626;margin-top:6px">Risk of inaction: $969M+ COGS at 95th percentile</p>
    </div>
    <div class="action-card" style="border-left:4px solid #dc2626">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-size:11px;font-weight:700;background:#fef2f2;color:#b91c1c;padding:2px 8px;border-radius:12px">P1 — IMMEDIATE</span>
        <span style="font-size:12px;font-weight:700">Lithium</span>
      </div>
      <p style="font-size:12px;color:#374151"><strong>Signal:</strong> +131.7% YoY · +57.6% above 7yr mean · 32% vol</p>
      <p style="font-size:12px;color:#374151;margin-top:4px"><strong>Action:</strong> Execute 6-month forward hedge — lock 50% of $1,530M exposure. Reopen supplier index clauses.</p>
      <p style="font-size:11px;color:#dc2626;margin-top:6px">Risk of inaction: $811M+ COGS at 95th percentile</p>
    </div>
    <div class="action-card" style="border-left:4px solid #dc2626">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-size:11px;font-weight:700;background:#fef2f2;color:#b91c1c;padding:2px 8px;border-radius:12px">P1 — IMMEDIATE</span>
        <span style="font-size:12px;font-weight:700">Aluminum</span>
      </div>
      <p style="font-size:12px;color:#374151"><strong>Signal:</strong> +135.1% YoY · +78.0% above 7yr mean · 62% vol (highest uncertainty)</p>
      <p style="font-size:12px;color:#374151;margin-top:4px"><strong>Action:</strong> Collar structures (cap + floor) for 50% of $1,020M exposure. Fixed forwards too expensive at 62% vol.</p>
      <p style="font-size:11px;color:#dc2626;margin-top:6px">Risk of inaction: $1,049M+ COGS at 95th percentile</p>
    </div>
    <div class="action-card" style="border-left:4px solid #16a34a">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-size:11px;font-weight:700;background:#f0fdf4;color:#15803d;padding:2px 8px;border-radius:12px">P1 — OPPORTUNITY</span>
        <span style="font-size:12px;font-weight:700">Natural Gas</span>
      </div>
      <p style="font-size:12px;color:#374151"><strong>Signal:</strong> −12.6% YoY · −12.5% below 7yr mean · Near Fair Value</p>
      <p style="font-size:12px;color:#374151;margin-top:4px"><strong>Action:</strong> Lock 12–24M supply contracts now. $42M annual saving vs. historical average.</p>
      <p style="font-size:11px;color:#16a34a;margin-top:6px">Rare window — mean reversion expected within 6–9 months</p>
    </div>
    <div class="action-card" style="border-left:4px solid #16a34a">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-size:11px;font-weight:700;background:#f0fdf4;color:#15803d;padding:2px 8px;border-radius:12px">P1 — OPPORTUNITY</span>
        <span style="font-size:12px;font-weight:700">Rhodium</span>
      </div>
      <p style="font-size:12px;color:#374151"><strong>Signal:</strong> −14.1% YoY · −13.7% below 7yr mean · Below Average</p>
      <p style="font-size:12px;color:#374151;margin-top:4px"><strong>Action:</strong> Extend supply contracts at current prices. $23M annual saving vs. historical average.</p>
      <p style="font-size:11px;color:#16a34a;margin-top:6px">SA supply risk makes timing critical</p>
    </div>
    <div class="action-card" style="border-left:4px solid #f59e0b">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-size:11px;font-weight:700;background:#fffbeb;color:#92400e;padding:2px 8px;border-radius:12px">P3 — STRUCTURAL</span>
        <span style="font-size:12px;font-weight:700">Operating Model</span>
      </div>
      <p style="font-size:12px;color:#374151">Monthly re-forecast cadence · Board-approved hedge ratio policy (40–60% metals, 20–30% battery materials) · Supplier indexation clauses · Quarterly EV-commodity review</p>
      <p style="font-size:11px;color:#d97706;margin-top:6px">Portfolio vol 25.7% pa → ±$2,184M 1-sigma annual cost uncertainty</p>
    </div>
  </div>

  <!-- Footer -->
  <div class="footer">
    <div>GIC Plan-to-Perform Engine v0.3.0 · Data: Yahoo Finance + FRED + O-U Model</div>
    <div>Generated: ${reportDate} ${reportTime} · CONFIDENTIAL — Board / C-Suite Only</div>
  </div>
</div>
</body>
</html>`;
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
  const [downloading, setDownloading] = useState(false);
  const { snapshot } = useRealtimeContext();

  useEffect(() => {
    let mounted = true;
    const fetchData = async () => {
      try {
        const data = await gicApi.getAnnualPnL();
        if (mounted) {
          setKpi(data);
          if (data.segments && data.segments.length > 0) {
            // Backend returns revenue in USD; normalise field names and convert to GBP.
            // Handles both old {name, revenue} shape and corrected {segment, revenue, volume} shape.
            const normalised = data.segments.map((s, i) => ({
              segment: s.segment || s.name || `Segment ${i + 1}`,
              revenue: Math.round((s.revenue || 0) / 1.27),
              volume: s.volume || 0,
            }));
            setSegments(normalised);
          }
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

  const handleDownloadReport = () => {
    setDownloading(true);
    try {
      const html = buildReportHTML({
        kpi: safeKpi,
        segments,
        ew,
        insightSummary,
        topInsights,
      });
      const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `GIC-Executive-Report-${new Date().toISOString().split('T')[0]}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className="text-blue-400">ℹ️</span>
        Connect backend:{' '}
        <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code>
        {' '}— showing mock data while offline.
      </div>

      {/* Page title + download */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Executive Summary</h2>
          <p className="text-slate-400 text-sm">GIC Plan-to-Perform · FY 2026 Consolidated View</p>
        </div>
        <button
          onClick={handleDownloadReport}
          disabled={downloading}
          className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all"
          style={{
            background: downloading ? '#334155' : 'linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%)',
            color: '#fff',
            border: '1px solid rgba(255,255,255,0.1)',
            cursor: downloading ? 'not-allowed' : 'pointer',
            boxShadow: downloading ? 'none' : '0 2px 8px rgba(29,78,216,0.4)',
          }}
        >
          {downloading ? (
            <>
              <svg className="animate-spin" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
              </svg>
              Generating…
            </>
          ) : (
            <>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
              Download Intelligence Report
            </>
          )}
        </button>
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
