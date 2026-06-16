import React, { useState, useEffect, useMemo } from 'react';
import InsightCard from '../components/insights/InsightCard';
import RecommendationPanel from '../components/insights/RecommendationPanel';
import Loading from '../components/common/Loading';
import { gicApi } from '../api/client';

// ── Mock fallback: 8 rich, realistic insights ────────────────────────────────
const MOCK_FEED = {
  insights: [
    {
      id: 'INS-001',
      category: 'Commodity',
      severity: 'critical',
      priority: 1,
      title: 'Lithium carbonate spike threatens EV battery margin',
      finding: 'Lithium is tracking +12.3% above the FY26 plan assumption, lifting EV battery pack cost by an estimated £74M against budget.',
      reasoning:
        'SARIMAX nowcast plus spot feed show a sustained breakout above the $14,200/t plan anchor. Elasticity model maps a 1% input move to ~£6.0M BOM impact for the EV segment given current volume.',
      impact_gbp: -74000000,
      impact_label: '-£74.0M',
      confidence: 0.82,
      recommended_action: 'Execute the pre-approved 6-month lithium hedge tranche and re-open supplier index clauses for Q3.',
      expected_action_savings_gbp: 41000000,
      affected_segments: ['EV', 'Performance'],
      supporting_metrics: { variance_pct: 12.3, plan_price: 14200, spot_price: 15950, mape: 9.1 },
    },
    {
      id: 'INS-002',
      category: 'Warranty',
      severity: 'critical',
      priority: 1,
      title: 'EV battery failure rate trending above accrual',
      finding: 'Field failure frequency on 2024-build EV packs is rising; projected 12M warranty cost exceeds the booked accrual by £18M.',
      reasoning:
        'Weibull hazard fit on field returns shows accelerating early-life failures. Accrual adequacy model flags a 7.4% shortfall versus the modelled liability at 90% confidence.',
      impact_gbp: -18000000,
      impact_label: '-£18.0M',
      confidence: 0.71,
      recommended_action: 'Top up the warranty accrual by £18M and launch an 8D root-cause on the cell supplier batch.',
      expected_action_savings_gbp: 12000000,
      affected_segments: ['EV'],
      supporting_metrics: { shortfall_pct: 7.4, claims_qoq: 19.0, dominant_mode: 'Battery' },
    },
    {
      id: 'INS-003',
      category: 'FX',
      severity: 'info',
      priority: 3,
      title: 'GBP weakness providing margin tailwind',
      finding: 'GBP/USD softening adds an estimated £22M of translation benefit across the USD-denominated commodity basket this quarter.',
      reasoning:
        'FX pass-through model attributes the favourable move to GBP/USD at 1.27 vs the 1.31 plan rate. Benefit partially offsets commodity headwinds.',
      impact_gbp: 22000000,
      impact_label: '+£22.0M',
      confidence: 0.68,
      recommended_action: 'Lock 50% of the realised FX tailwind via forward contracts to protect H2 guidance.',
      expected_action_savings_gbp: 11000000,
      affected_segments: ['Luxury SUV', 'Premium SUV'],
      supporting_metrics: { plan_rate: 1.31, spot_rate: 1.272, exposure_usd: 980000000 },
    },
    {
      id: 'INS-004',
      category: 'Demand',
      severity: 'warning',
      priority: 2,
      title: 'Premium SUV order intake softening in EU',
      finding: 'EU order bank for Premium SUV is down 6% QoQ, putting £38M of planned H2 revenue at risk if the trend persists.',
      reasoning:
        'Demand sensing model detects a regime shift (Hurst 0.61) in EU registrations; macro PMI deterioration correlates at 0.74.',
      impact_gbp: -38000000,
      impact_label: '-£38.0M',
      confidence: 0.64,
      recommended_action: 'Reallocate marketing spend to resilient UK/US channels and trigger a tactical finance offer in EU.',
      expected_action_savings_gbp: 16000000,
      affected_segments: ['Premium SUV'],
      supporting_metrics: { order_qoq: -6.0, pmi: 47.2, hurst: 0.61 },
    },
    {
      id: 'INS-005',
      category: 'Commodity',
      severity: 'warning',
      priority: 2,
      title: 'Aluminium softening — procurement timing opportunity',
      finding: 'Aluminium is forecast to ease 4-6% over the next quarter, opening a £19M cost-down opportunity on body structures.',
      reasoning:
        'Improved supply signals and inventory builds drive the downward nowcast; calibrated 79% confidence over the 3-month horizon.',
      impact_gbp: 19000000,
      impact_label: '+£19.0M',
      confidence: 0.79,
      recommended_action: 'Defer non-critical aluminium purchase orders by 4-6 weeks to capture the dip.',
      expected_action_savings_gbp: 13000000,
      affected_segments: ['Luxury SUV', 'Premium SUV', 'Performance'],
      supporting_metrics: { forecast_change_pct: -5.0, confidence: 79 },
    },
    {
      id: 'INS-006',
      category: 'Energy',
      severity: 'warning',
      priority: 2,
      title: 'Natural gas volatility breaches governance threshold',
      finding: 'Natural gas forecast MAPE of 31% exceeds the 20% high-volatility governance threshold, adding £9M of unhedged plant energy risk.',
      reasoning:
        'Model error has widened beyond the control band, mandating scenario-based planning rather than point forecasts for H2 energy.',
      impact_gbp: -9000000,
      impact_label: '-£9.0M',
      confidence: 0.58,
      recommended_action: 'Switch H2 energy budgeting to scenario bands and secure a fixed-price strip for baseload.',
      expected_action_savings_gbp: 5000000,
      affected_segments: ['Manufacturing'],
      supporting_metrics: { mape: 31.0, threshold: 20.0 },
    },
    {
      id: 'INS-007',
      category: 'Margin',
      severity: 'info',
      priority: 3,
      title: 'Performance segment mix enrichment lifting margin',
      finding: 'Richer option take-rate in the Performance segment is adding ~60bps of gross margin, worth £14M vs plan.',
      reasoning:
        'Price/mix attribution shows higher-trim penetration; sustained over two quarters and resilient to discounting.',
      impact_gbp: 14000000,
      impact_label: '+£14.0M',
      confidence: 0.73,
      recommended_action: 'Codify the winning option bundles into the FY27 standard configuration to lock the mix benefit.',
      expected_action_savings_gbp: 8000000,
      affected_segments: ['Performance'],
      supporting_metrics: { margin_bps: 60, take_rate_pct: 38 },
    },
    {
      id: 'INS-008',
      category: 'Commodity',
      severity: 'info',
      priority: 4,
      title: 'Palladium regime change — rebalance model weights',
      finding: 'Palladium has entered a trending regime (Hurst 0.62); current ensemble under-weights the momentum signal.',
      reasoning:
        'Regime detector flags persistence; backtest suggests shifting ensemble weight toward XGBoost improves MAPE by ~3pts.',
      impact_gbp: 4000000,
      impact_label: '+£4.0M',
      confidence: 0.61,
      recommended_action: 'Rebalance the forecast ensemble toward the gradient-boosted model for PGM commodities.',
      expected_action_savings_gbp: 3000000,
      affected_segments: ['Powertrain'],
      supporting_metrics: { hurst: 0.62, mape_gain: 3.0 },
    },
  ],
  summary: {
    n_critical: 2,
    n_warning: 3,
    // Net: -74-18+22-38+19-9+14+4 = -80M  |  Upside (positives): 22+19+14+4 = 59M
    // Wtd confidence = Σ(|impact|×conf) / Σ|impact| = 145.63/198 ≈ 0.74
    total_impact_gbp: -80000000,
    total_opportunity_gbp: 59000000,
    weighted_confidence: 0.74,
  },
};

const CATEGORIES = ['All', 'Commodity', 'Warranty', 'FX', 'Demand', 'Energy', 'Margin'];
const SEVERITIES = ['All', 'critical', 'warning', 'info'];

function fmtGBP(v) {
  if (v === null || v === undefined) return '—';
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}£${(abs / 1e9).toFixed(2)}bn`;
  if (abs >= 1e6) return `${sign}£${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}£${(abs / 1e3).toFixed(0)}K`;
  return `${sign}£${abs.toFixed(0)}`;
}

function StatTile({ label, value, color }) {
  return (
    <div className="rounded-xl p-4 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
      <div className="text-[11px] text-slate-400 uppercase tracking-wide">{label}</div>
      <div className="text-2xl font-bold tabular-nums mt-1" style={{ color: color || '#ffffff' }}>{value}</div>
    </div>
  );
}

export default function InsightsCenter() {
  const [feed, setFeed] = useState(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(false);
  const [catFilter, setCatFilter] = useState('All');
  const [sevFilter, setSevFilter] = useState('All');
  const [sortBy, setSortBy] = useState('priority'); // 'priority' | 'impact'

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const data = await gicApi.insightsFeed();
        if (mounted && data?.insights) {
          setFeed(data);
          setLive(true);
        } else if (mounted) {
          setFeed(MOCK_FEED);
        }
      } catch {
        if (mounted) setFeed(MOCK_FEED);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const data = feed || MOCK_FEED;
  const summary = data.summary || MOCK_FEED.summary;

  const filtered = useMemo(() => {
    let list = (data.insights || []).filter((i) => {
      if (catFilter !== 'All' && i.category !== catFilter) return false;
      if (sevFilter !== 'All' && i.severity !== sevFilter) return false;
      return true;
    });
    if (sortBy === 'priority') list = [...list].sort((a, b) => (a.priority ?? 9) - (b.priority ?? 9));
    else list = [...list].sort((a, b) => Math.abs(b.impact_gbp ?? 0) - Math.abs(a.impact_gbp ?? 0));
    return list;
  }, [data.insights, catFilter, sevFilter, sortBy]);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className={`w-2 h-2 rounded-full ${live ? 'bg-emerald-400' : 'bg-amber-400'} animate-pulse`} />
        {live ? (
          <span className="text-emerald-300">Connected to live insights feed (<code>/insights/feed</code>).</span>
        ) : (
          <span>Showing high-fidelity mock insights — start backend: <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code></span>
        )}
      </div>

      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Insights Center</h2>
        <p className="text-slate-400 text-sm">Prescriptive, quantified intelligence — ranked by financial impact and confidence.</p>
      </div>

      {/* Summary stat row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <StatTile label="Critical" value={summary.n_critical} color="#f87171" />
        <StatTile label="Warnings" value={summary.n_warning} color="#fbbf24" />
        <StatTile label="Net Impact" value={fmtGBP(summary.total_impact_gbp)} color={(summary.total_impact_gbp ?? 0) >= 0 ? '#34d399' : '#f87171'} />
        <StatTile label="Total Upside" value={fmtGBP(summary.total_opportunity_gbp)} color="#34d399" />
        <StatTile label="Wtd. Confidence" value={`${Math.round((summary.weighted_confidence ?? 0) * 100)}%`} color="#60a5fa" />
      </div>

      {/* Filters */}
      <div className="rounded-xl p-4 border border-slate-700 flex flex-wrap items-center gap-x-6 gap-y-3" style={{ backgroundColor: '#1e293b' }}>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[11px] text-slate-500 uppercase">Category</span>
          {CATEGORIES.map((c) => (
            <button
              key={c}
              onClick={() => setCatFilter(c)}
              className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${catFilter === c ? 'bg-blue-500/20 border-blue-500 text-blue-300' : 'border-slate-600 text-slate-400 hover:text-slate-200'}`}
            >
              {c}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[11px] text-slate-500 uppercase">Severity</span>
          {SEVERITIES.map((s) => (
            <button
              key={s}
              onClick={() => setSevFilter(s)}
              className={`text-xs px-2.5 py-1 rounded-full border capitalize transition-colors ${sevFilter === s ? 'bg-blue-500/20 border-blue-500 text-blue-300' : 'border-slate-600 text-slate-400 hover:text-slate-200'}`}
            >
              {s}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-[11px] text-slate-500 uppercase">Sort</span>
          {[['priority', 'Priority'], ['impact', 'Impact']].map(([k, lbl]) => (
            <button
              key={k}
              onClick={() => setSortBy(k)}
              className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${sortBy === k ? 'bg-blue-500/20 border-blue-500 text-blue-300' : 'border-slate-600 text-slate-400 hover:text-slate-200'}`}
            >
              {lbl}
            </button>
          ))}
        </div>
      </div>

      {/* Main grid: insights + recommendation rail */}
      {loading ? (
        <Loading message="Loading insights…" />
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-2 space-y-4">
            {filtered.length === 0 ? (
              <div className="rounded-xl p-10 text-center border border-slate-700 text-slate-500" style={{ backgroundColor: '#1e293b' }}>
                No insights match the current filters.
              </div>
            ) : (
              filtered.map((ins) => <InsightCard key={ins.id} insight={ins} />)
            )}
          </div>
          <div className="xl:col-span-1">
            <div className="xl:sticky xl:top-6">
              <RecommendationPanel insights={data.insights || []} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
