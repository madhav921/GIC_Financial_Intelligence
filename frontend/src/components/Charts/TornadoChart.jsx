import React from 'react';
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

/**
 * Tornado / sensitivity chart. Each driver shows its low- and high-shock impact
 * on the target metric as a horizontal bar diverging from zero. Bars are sorted
 * by total swing (widest at top) — the canonical tornado layout.
 *
 * Props:
 *  - items: [{ name, low (Δ£M at downside shock), high (Δ£M at upside shock) }]
 *  - unit: label suffix (default 'M')
 */
const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload || {};
  return (
    <div className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl" style={{ backgroundColor: '#0f172a' }}>
      <p className="text-slate-200 font-medium mb-1">{label}</p>
      <p className="text-emerald-400">Upside: {d.high >= 0 ? '+' : ''}£{Math.round(d.high)}M</p>
      <p className="text-red-400">Downside: {d.low >= 0 ? '+' : ''}£{Math.round(d.low)}M</p>
      <p className="text-slate-400 mt-0.5">Swing: £{Math.round(Math.abs(d.high - d.low))}M</p>
    </div>
  );
};

export default function TornadoChart({ items = [], height = 220 }) {
  const data = [...items]
    .map((it) => ({ ...it, swing: Math.abs((it.high ?? 0) - (it.low ?? 0)) }))
    .sort((a, b) => a.swing - b.swing); // recharts renders bottom-up; widest ends on top

  const maxAbs = Math.max(1, ...data.flatMap((d) => [Math.abs(d.low), Math.abs(d.high)]));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart layout="vertical" data={data} margin={{ top: 4, right: 24, left: 24, bottom: 4 }} stackOffset="sign">
        <CartesianGrid stroke="#334155" strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          domain={[-maxAbs * 1.1, maxAbs * 1.1]}
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: '#475569' }}
          tickLine={false}
          tickFormatter={(v) => `£${Math.round(v)}M`}
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fill: '#cbd5e1', fontSize: 12 }}
          axisLine={false}
          tickLine={false}
          width={96}
        />
        <Tooltip content={<Tip />} cursor={{ fill: 'rgba(148,163,184,0.06)' }} />
        <ReferenceLine x={0} stroke="#64748b" />
        <Bar dataKey="low" stackId="t" radius={[2, 2, 2, 2]} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.low < 0 ? '#ef4444' : '#22c55e'} />
          ))}
        </Bar>
        <Bar dataKey="high" stackId="t" radius={[2, 2, 2, 2]} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.high < 0 ? '#ef4444' : '#22c55e'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
