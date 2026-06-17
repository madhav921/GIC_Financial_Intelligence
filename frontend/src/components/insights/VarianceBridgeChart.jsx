import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';

function fmtGBP(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}£${(abs / 1e9).toFixed(2)}bn`;
  if (abs >= 1e6) return `${sign}£${(abs / 1e6).toFixed(0)}M`;
  if (abs >= 1e3) return `${sign}£${(abs / 1e3).toFixed(0)}K`;
  return `${sign}£${abs.toFixed(0)}`;
}

const FAVORABLE = '#22c55e';
const ADVERSE = '#ef4444';
const TOTAL = '#3b82f6';

/**
 * Builds floating-waterfall rows. Each row has:
 *  base  — invisible offset bar
 *  delta — visible coloured bar (absolute height of the step)
 *  type  — 'start' | 'favorable' | 'adverse' | 'end'
 */
function buildWaterfall(planEbit, actualEbit, bridge) {
  const rows = [];
  let running = planEbit;

  rows.push({
    name: 'Plan EBIT',
    base: 0,
    delta: planEbit,
    value: planEbit,
    type: 'start',
    explanation: 'Baseline planned EBIT for the period.',
  });

  bridge.forEach((d) => {
    const delta = d.delta_gbp;
    const start = running;
    const end = running + delta;
    const base = Math.min(start, end);
    rows.push({
      name: d.driver,
      base,
      delta: Math.abs(delta),
      value: delta,
      pct: d.pct,
      type: d.direction === 'favorable' ? 'favorable' : 'adverse',
      explanation: d.explanation,
      running: end,
    });
    running = end;
  });

  rows.push({
    name: 'Actual EBIT',
    base: 0,
    delta: actualEbit,
    value: actualEbit,
    type: 'end',
    explanation: 'Realised actual EBIT after all drivers.',
  });

  return rows;
}

const COLOR_BY_TYPE = {
  start: '#64748b',
  favorable: FAVORABLE,
  adverse: ADVERSE,
  end: TOTAL,
};

const WaterfallTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  const signed = row.type === 'favorable' ? '+' : row.type === 'adverse' ? '-' : '';
  const valColor =
    row.type === 'favorable' ? FAVORABLE : row.type === 'adverse' ? ADVERSE : '#cbd5e1';
  return (
    <div className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl max-w-xs" style={{ backgroundColor: '#0f172a' }}>
      <p className="text-slate-200 font-semibold mb-1">{row.name}</p>
      <div className="flex items-center justify-between gap-6 mb-1">
        <span className="text-slate-400">{row.type === 'start' || row.type === 'end' ? 'EBIT' : 'Impact'}</span>
        <span className="font-mono font-bold" style={{ color: valColor }}>
          {signed}
          {fmtGBP(Math.abs(row.value))}
        </span>
      </div>
      {row.pct !== undefined && (
        <div className="flex items-center justify-between gap-6 mb-1">
          <span className="text-slate-400">% of plan</span>
          <span className="font-mono text-slate-200">{row.pct > 0 ? '+' : ''}{Number(row.pct).toFixed(1)}%</span>
        </div>
      )}
      {row.explanation && (
        <p className="text-slate-400 leading-snug border-t border-slate-700 pt-1.5 mt-1.5">{row.explanation}</p>
      )}
    </div>
  );
};

export default function VarianceBridgeChart({ planEbit, actualEbit, bridge = [], height = 360 }) {
  const data = useMemo(
    () => buildWaterfall(planEbit, actualEbit, bridge),
    [planEbit, actualEbit, bridge]
  );

  return (
    <div className="rounded-xl border border-slate-700 p-6" style={{ backgroundColor: '#1e293b' }}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">EBIT Variance Bridge</h3>
          <p className="text-xs text-slate-400 mt-0.5">Plan → Actual decomposition by driver</p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: FAVORABLE }} /> Favorable
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: ADVERSE }} /> Adverse
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: TOTAL }} /> Total
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 16, right: 16, left: 8, bottom: 48 }} barCategoryGap="22%">
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: '#475569' }}
            tickLine={false}
            angle={-30}
            textAnchor="end"
            interval={0}
            height={50}
          />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={60}
            tickFormatter={(v) => fmtGBP(v)}
            domain={[0, 'auto']}
          />
          <Tooltip content={<WaterfallTooltip />} cursor={{ fill: 'rgba(148,163,184,0.06)' }} />
          <ReferenceLine y={planEbit} stroke="#475569" strokeDasharray="4 4" />
          {/* invisible base to float the bars */}
          <Bar dataKey="base" stackId="wf" fill="transparent" isAnimationActive={false} />
          <Bar dataKey="delta" stackId="wf" radius={[3, 3, 0, 0]} isAnimationActive={false}>
            {data.map((row, i) => (
              <Cell key={i} fill={COLOR_BY_TYPE[row.type] || '#64748b'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export { fmtGBP as fmtGBPBridge };
