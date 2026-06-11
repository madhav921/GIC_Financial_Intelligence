import React, { useMemo, useState } from 'react';

function fmtGBP(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}£${(abs / 1e9).toFixed(2)}bn`;
  if (abs >= 1e6) return `${sign}£${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}£${(abs / 1e3).toFixed(0)}K`;
  return `${sign}£${abs.toLocaleString()}`;
}

const PRIORITY_COLOR = {
  1: { bg: 'rgba(239,68,68,0.15)', text: '#f87171', border: '#dc2626' },
  2: { bg: 'rgba(245,158,11,0.15)', text: '#fbbf24', border: '#d97706' },
  3: { bg: 'rgba(59,130,246,0.15)', text: '#60a5fa', border: '#2563eb' },
};

export default function RecommendationPanel({ insights = [] }) {
  const [sort, setSort] = useState('savings'); // 'savings' | 'priority'

  const actions = useMemo(() => {
    const list = insights
      .filter((i) => i.recommended_action)
      .map((i) => ({
        id: i.id,
        action: i.recommended_action,
        savings: i.expected_action_savings_gbp || 0,
        priority: i.priority ?? 5,
        category: i.category,
        severity: i.severity,
      }));
    if (sort === 'savings') list.sort((a, b) => b.savings - a.savings);
    else list.sort((a, b) => a.priority - b.priority || b.savings - a.savings);
    return list;
  }, [insights, sort]);

  const totalUpside = useMemo(
    () => actions.reduce((s, a) => s + (a.savings || 0), 0),
    [actions]
  );

  return (
    <div className="rounded-xl border border-slate-700 p-6" style={{ backgroundColor: '#1e293b' }}>
      <div className="flex items-start justify-between gap-3 mb-1">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">Recommended Actions</h3>
          <p className="text-xs text-slate-400 mt-0.5">Prioritised play list across all insights</p>
        </div>
      </div>

      {/* Total upside header */}
      <div
        className="mt-3 mb-4 rounded-lg p-4 border border-emerald-800/50"
        style={{ background: 'linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.03))' }}
      >
        <div className="text-[11px] uppercase tracking-wide text-emerald-400 font-semibold">
          Total addressable upside
        </div>
        <div className="text-3xl font-bold text-emerald-300 tabular-nums mt-1">{fmtGBP(totalUpside)}</div>
        <div className="text-xs text-slate-400 mt-1">{actions.length} actionable recommendations</div>
      </div>

      {/* Sort toggle */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-[11px] text-slate-500 uppercase">Sort</span>
        {['savings', 'priority'].map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSort(s)}
            className={`text-xs px-2.5 py-1 rounded-full border transition-colors ${
              sort === s
                ? 'bg-blue-500/20 border-blue-500 text-blue-300'
                : 'border-slate-600 text-slate-400 hover:text-slate-200'
            }`}
          >
            {s === 'savings' ? 'By savings' : 'By priority'}
          </button>
        ))}
      </div>

      <div className="space-y-2 max-h-[520px] overflow-y-auto scrollbar-thin pr-1">
        {actions.length === 0 && (
          <p className="text-sm text-slate-500 py-6 text-center">No recommended actions available.</p>
        )}
        {actions.map((a, idx) => {
          const pc = PRIORITY_COLOR[a.priority] || PRIORITY_COLOR[3];
          return (
            <div
              key={a.id || idx}
              className="rounded-lg p-3 border border-slate-700 hover:border-slate-500 transition-colors"
              style={{ backgroundColor: '#0f172a' }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2.5 min-w-0">
                  <span className="text-slate-500 text-xs font-mono mt-0.5 flex-shrink-0">{idx + 1}.</span>
                  <p className="text-sm text-slate-200 leading-snug">{a.action}</p>
                </div>
                <span
                  className="text-[10px] font-bold px-1.5 py-0.5 rounded border flex-shrink-0"
                  style={{ backgroundColor: pc.bg, color: pc.text, borderColor: pc.border }}
                >
                  P{a.priority}
                </span>
              </div>
              <div className="flex items-center justify-between mt-2 pl-6">
                {a.category && <span className="text-[11px] text-slate-500">{a.category}</span>}
                {a.savings > 0 && (
                  <span className="text-sm font-bold text-emerald-300 tabular-nums">{fmtGBP(a.savings)}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
