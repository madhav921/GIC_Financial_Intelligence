import React, { useState } from 'react';

// ── Local formatting helpers ─────────────────────────────────────────────────
function fmtGBP(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}£${(abs / 1e9).toFixed(2)}bn`;
  if (abs >= 1e6) return `${sign}£${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}£${(abs / 1e3).toFixed(0)}K`;
  return `${sign}£${abs.toLocaleString()}`;
}

const SEVERITY = {
  critical: { stripe: '#ef4444', badge: 'red', label: 'Critical', dot: '#f87171' },
  warning: { stripe: '#f59e0b', badge: 'yellow', label: 'Warning', dot: '#fbbf24' },
  info: { stripe: '#3b82f6', badge: 'blue', label: 'Info', dot: '#60a5fa' },
};

const BADGE_STYLE = {
  red: { bg: 'rgba(239,68,68,0.15)', border: '#dc2626', text: '#f87171' },
  yellow: { bg: 'rgba(245,158,11,0.15)', border: '#d97706', text: '#fbbf24' },
  blue: { bg: 'rgba(59,130,246,0.15)', border: '#2563eb', text: '#60a5fa' },
  slate: { bg: 'rgba(100,116,139,0.15)', border: '#475569', text: '#94a3b8' },
};

function Chip({ children, color = 'slate' }) {
  const s = BADGE_STYLE[color] || BADGE_STYLE.slate;
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold border whitespace-nowrap"
      style={{ backgroundColor: s.bg, borderColor: s.border, color: s.text }}
    >
      {children}
    </span>
  );
}

export default function InsightCard({ insight, defaultExpanded = false }) {
  const [open, setOpen] = useState(defaultExpanded);
  if (!insight) return null;

  const sev = SEVERITY[insight.severity] || SEVERITY.info;
  const impact = insight.impact_gbp || 0;
  const isOpportunity = impact >= 0;
  const conf = Math.round((insight.confidence ?? 0) * 100);
  const savings = insight.expected_action_savings_gbp || 0;
  const segments = insight.affected_segments || [];
  const metrics = insight.supporting_metrics || {};

  return (
    <div
      className="rounded-xl border border-slate-700 overflow-hidden transition-all duration-200 hover:border-slate-500 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-black/30"
      style={{ backgroundColor: '#1e293b' }}
    >
      <div className="flex">
        {/* Severity stripe */}
        <div className="w-1.5 flex-shrink-0" style={{ backgroundColor: sev.stripe }} />

        <div className="flex-1 p-5 min-w-0">
          {/* Header row */}
          <div className="flex items-start justify-between gap-3 mb-2">
            <div className="flex items-center gap-2 flex-wrap">
              <Chip color={sev.badge}>
                <span className="mr-1" style={{ color: sev.dot }}>●</span>
                {sev.label}
              </Chip>
              {insight.category && <Chip color="slate">{insight.category}</Chip>}
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-bold bg-slate-900 text-slate-300 border border-slate-600">
                P{insight.priority ?? '—'}
              </span>
            </div>
            <div className="text-right flex-shrink-0">
              <div
                className="text-xl font-bold tabular-nums leading-none"
                style={{ color: isOpportunity ? '#34d399' : '#f87171' }}
              >
                {insight.impact_label || fmtGBP(impact)}
              </div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wide mt-1">
                {isOpportunity ? 'Upside' : 'Exposure'}
              </div>
            </div>
          </div>

          {/* Title + finding */}
          <h3 className="text-base font-bold text-white leading-snug mb-1">{insight.title}</h3>
          <p className="text-sm text-slate-300 leading-relaxed">{insight.finding}</p>

          {/* Confidence meter */}
          <div className="mt-4">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[11px] text-slate-400 uppercase tracking-wide">Model confidence</span>
              <span className="text-xs font-mono font-semibold text-slate-200">{conf}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-slate-700 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${conf}%`,
                  background:
                    conf >= 75
                      ? 'linear-gradient(90deg,#22c55e,#34d399)'
                      : conf >= 50
                      ? 'linear-gradient(90deg,#f59e0b,#fbbf24)'
                      : 'linear-gradient(90deg,#ef4444,#f87171)',
                }}
              />
            </div>
          </div>

          {/* Expandable "Why" */}
          {insight.reasoning && (
            <button
              type="button"
              onClick={() => setOpen((o) => !o)}
              className="mt-4 flex items-center gap-1.5 text-xs font-semibold text-blue-400 hover:text-blue-300 transition-colors"
            >
              <span className={`transition-transform duration-200 ${open ? 'rotate-90' : ''}`}>▶</span>
              Why this matters
            </button>
          )}
          {open && insight.reasoning && (
            <div className="mt-2 rounded-lg p-3 border border-slate-700 text-xs text-slate-300 leading-relaxed" style={{ backgroundColor: '#0f172a' }}>
              {insight.reasoning}
              {Object.keys(metrics).length > 0 && (
                <div className="mt-3 grid grid-cols-2 gap-2">
                  {Object.entries(metrics).map(([k, v]) => (
                    <div key={k} className="flex items-center justify-between gap-2 rounded bg-slate-800/60 px-2 py-1">
                      <span className="text-slate-500 capitalize truncate">{k.replace(/_/g, ' ')}</span>
                      <span className="font-mono text-slate-200 font-semibold">
                        {typeof v === 'number' ? v.toLocaleString() : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Recommended action */}
          {insight.recommended_action && (
            <div
              className="mt-4 rounded-lg p-3 border border-emerald-800/50 flex items-start gap-3"
              style={{ backgroundColor: 'rgba(16,185,129,0.07)' }}
            >
              <span className="text-emerald-400 mt-0.5 flex-shrink-0">➜</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-3 mb-0.5">
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-emerald-400">
                    Recommended action
                  </span>
                  {savings > 0 && (
                    <span className="text-sm font-bold text-emerald-300 tabular-nums whitespace-nowrap">
                      {fmtGBP(savings)} <span className="text-[10px] text-emerald-500 font-medium">est. savings</span>
                    </span>
                  )}
                </div>
                <p className="text-sm text-slate-200 leading-snug">{insight.recommended_action}</p>
                {segments.length > 0 && (
                  <div className="flex items-center gap-1.5 flex-wrap mt-2">
                    <span className="text-[10px] text-slate-500 uppercase">Affects:</span>
                    {segments.map((s) => (
                      <span key={s} className="text-[11px] px-1.5 py-0.5 rounded bg-slate-700/70 text-slate-300">
                        {s}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export { fmtGBP };
