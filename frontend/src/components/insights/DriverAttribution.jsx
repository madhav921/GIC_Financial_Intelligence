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

const POSITIVE = '#ef4444'; // contributes to risk / adverse → red-ish
const NEGATIVE = '#22c55e'; // reduces risk / favorable → green-ish

const AttribTooltip = ({ active, payload, valueLabel }) => {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload;
  if (!row) return null;
  const positive = row.value >= 0;
  return (
    <div className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl" style={{ backgroundColor: '#0f172a' }}>
      <p className="text-slate-200 font-semibold mb-1">{row.name}</p>
      <div className="flex items-center justify-between gap-6">
        <span className="text-slate-400">{valueLabel}</span>
        <span className="font-mono font-bold" style={{ color: positive ? POSITIVE : NEGATIVE }}>
          {positive ? '+' : ''}{Number(row.value).toFixed(2)}
        </span>
      </div>
    </div>
  );
};

/**
 * Horizontal diverging bar chart of signed driver contributions.
 * Props:
 *   drivers: [{ name, value }]  (value signed)
 *   valueLabel: tooltip label (default 'Contribution')
 *   positiveLabel / negativeLabel: legend captions
 */
export default function DriverAttribution({
  drivers = [],
  height = 240,
  valueLabel = 'Contribution',
  positiveLabel = 'Increases risk',
  negativeLabel = 'Reduces risk',
}) {
  const data = useMemo(
    () => [...drivers].sort((a, b) => b.value - a.value),
    [drivers]
  );

  const maxAbs = useMemo(
    () => Math.max(1, ...data.map((d) => Math.abs(d.value))),
    [data]
  );

  return (
    <div>
      <div className="flex items-center gap-4 text-xs mb-3">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: NEGATIVE }} /> {negativeLabel}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: POSITIVE }} /> {positiveLabel}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
          barCategoryGap="28%"
        >
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={[-maxAbs * 1.1, maxAbs * 1.1]}
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: '#475569' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: '#cbd5e1', fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            width={110}
          />
          <Tooltip
            content={<AttribTooltip valueLabel={valueLabel} />}
            cursor={{ fill: 'rgba(148,163,184,0.06)' }}
          />
          <ReferenceLine x={0} stroke="#64748b" />
          <Bar dataKey="value" radius={[3, 3, 3, 3]} isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.value >= 0 ? POSITIVE : NEGATIVE} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
