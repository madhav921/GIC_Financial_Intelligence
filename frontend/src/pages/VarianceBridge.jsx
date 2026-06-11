import React, { useState, useEffect, useMemo } from 'react';
import VarianceBridgeChart from '../components/insights/VarianceBridgeChart';
import KPICard from '../components/Charts/KPICard';
import Badge from '../components/common/Badge';
import Loading from '../components/common/Loading';
import { gicApi } from '../api/client';

// ── Mock fallback ─────────────────────────────────────────────────────────────
const MOCK_BRIDGE = {
  plan_ebit: 1400000000,
  actual_ebit: 1401000000,
  total_variance_gbp: 1000000,
  total_variance_pct: 0.07,
  bridge: [
    { driver: 'Volume', delta_gbp: 64000000, pct: 4.6, direction: 'favorable', explanation: 'Luxury SUV volume recovery added units above plan, lifting contribution margin.' },
    { driver: 'Price/Mix', delta_gbp: 38000000, pct: 2.7, direction: 'favorable', explanation: 'Richer trim and option take-rate in Performance enriched the mix.' },
    { driver: 'Commodity', delta_gbp: -86000000, pct: -6.1, direction: 'adverse', explanation: 'Lithium and cobalt spikes raised BOM cost on EV and Performance lines.' },
    { driver: 'FX', delta_gbp: 22000000, pct: 1.6, direction: 'favorable', explanation: 'GBP weakness vs USD delivered a translation tailwind on the commodity basket.' },
    { driver: 'Warranty', delta_gbp: -28000000, pct: -2.0, direction: 'adverse', explanation: 'EV battery field failures drove an accrual top-up above plan.' },
    { driver: 'Other', delta_gbp: -9000000, pct: -0.6, direction: 'adverse', explanation: 'Energy volatility and miscellaneous SG&A timing.' },
  ],
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

export default function VarianceBridge() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(false);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await gicApi.varianceBridge();
        if (mounted && res?.bridge) {
          setData(res);
          setLive(true);
        } else if (mounted) {
          setData(MOCK_BRIDGE);
        }
      } catch {
        if (mounted) setData(MOCK_BRIDGE);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const d = data || MOCK_BRIDGE;
  const variancePositive = (d.total_variance_gbp ?? 0) >= 0;

  const largestAdverse = useMemo(() => {
    const adverse = (d.bridge || []).filter((b) => b.direction === 'adverse');
    if (!adverse.length) return null;
    return adverse.reduce((m, b) => (Math.abs(b.delta_gbp) > Math.abs(m.delta_gbp) ? b : m));
  }, [d.bridge]);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className={`w-2 h-2 rounded-full ${live ? 'bg-emerald-400' : 'bg-amber-400'} animate-pulse`} />
        {live ? (
          <span className="text-emerald-300">Connected to variance engine (<code>/insights/variance-bridge</code>).</span>
        ) : (
          <span>Showing mock variance bridge — start backend: <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code></span>
        )}
      </div>

      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-1">Plan-to-Perform · Variance Bridge</h2>
        <p className="text-slate-400 text-sm">
          Driver-based decomposition of Plan vs Actual EBIT — every gap explained by Volume, Price/Mix, Commodity, FX, Warranty and Other.
        </p>
      </div>

      {loading ? (
        <Loading message="Building variance bridge…" />
      ) : (
        <>
          {/* KPI row */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <KPICard title="Plan EBIT" value={d.plan_ebit} subtitle="Budgeted FY26" format="currency" />
            <KPICard title="Actual EBIT" value={d.actual_ebit} subtitle="Realised to date" format="currency" />
            <div className="rounded-xl p-5 border border-slate-700 flex flex-col gap-2" style={{ backgroundColor: '#1e293b' }}>
              <div className="text-xs font-medium text-slate-400 uppercase tracking-wide">Total Variance</div>
              <div className="text-3xl font-bold tabular-nums" style={{ color: variancePositive ? '#34d399' : '#f87171' }}>
                {variancePositive ? '+' : ''}{fmtGBP(d.total_variance_gbp)}
              </div>
              <div className="flex items-center gap-2 mt-1">
                <Badge label={`${variancePositive ? '+' : ''}${Number(d.total_variance_pct ?? 0).toFixed(2)}% vs plan`} color={variancePositive ? 'green' : 'red'} />
                <Badge label={variancePositive ? 'Favorable' : 'Adverse'} color={variancePositive ? 'green' : 'red'} />
              </div>
            </div>
          </div>

          {/* Waterfall centerpiece */}
          <VarianceBridgeChart planEbit={d.plan_ebit} actualEbit={d.actual_ebit} bridge={d.bridge} />

          {/* Driver table */}
          <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
            <h3 className="text-lg font-semibold text-slate-100 mb-4">Driver Detail</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left text-xs text-slate-400 font-medium pb-2">Driver</th>
                  <th className="text-right text-xs text-slate-400 font-medium pb-2">Δ EBIT</th>
                  <th className="text-right text-xs text-slate-400 font-medium pb-2">% of Plan</th>
                  <th className="text-center text-xs text-slate-400 font-medium pb-2">Direction</th>
                  <th className="text-left text-xs text-slate-400 font-medium pb-2 pl-4">Explanation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {(d.bridge || []).map((b) => {
                  const fav = b.direction === 'favorable';
                  return (
                    <tr key={b.driver} className="hover:bg-slate-700/30 transition-colors align-top">
                      <td className="py-3 text-slate-200 font-medium whitespace-nowrap">{b.driver}</td>
                      <td className="py-3 text-right font-mono font-semibold" style={{ color: fav ? '#34d399' : '#f87171' }}>
                        {fav ? '+' : ''}{fmtGBP(b.delta_gbp)}
                      </td>
                      <td className="py-3 text-right font-mono text-slate-300">
                        {b.pct > 0 ? '+' : ''}{Number(b.pct).toFixed(1)}%
                      </td>
                      <td className="py-3 text-center">
                        <Badge label={fav ? 'Favorable' : 'Adverse'} color={fav ? 'green' : 'red'} />
                      </td>
                      <td className="py-3 pl-4 text-slate-400 text-xs leading-snug max-w-md">{b.explanation}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Callout */}
          {largestAdverse && (
            <div className="rounded-xl p-6 border border-red-900/50" style={{ backgroundColor: 'rgba(239,68,68,0.06)' }}>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-xl">🎯</span>
                <h3 className="text-lg font-semibold text-slate-100">What this means · Recommended focus</h3>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">
                The largest adverse driver is <span className="font-semibold text-red-300">{largestAdverse.driver}</span> at{' '}
                <span className="font-mono font-bold text-red-300">{fmtGBP(largestAdverse.delta_gbp)}</span>{' '}
                ({Number(largestAdverse.pct).toFixed(1)}% of plan). {largestAdverse.explanation} Prioritise mitigation here:
                hedging, supplier renegotiation or scenario-based budgeting will deliver the highest marginal return on management attention,
                and recovering even half of this gap would more than offset the residual adverse drivers.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
