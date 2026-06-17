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
 * Probability distribution histogram for Monte Carlo outcomes with VaR / CVaR
 * markers and a mean reference line. Bars left of the VaR threshold are shaded
 * red (tail risk), the rest accent blue.
 *
 * Props:
 *  - bins: [{ x: number (bin center, £M), count: number }]
 *  - mean, var95, cvar95: numbers in £M
 *  - color: accent colour for the body of the distribution
 */
const Tip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload || {};
  return (
    <div className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl" style={{ backgroundColor: '#0f172a' }}>
      <p className="text-slate-300 font-medium mb-1">£{Math.round(d.x).toLocaleString()}M</p>
      <p className="text-slate-400">
        Frequency: <span className="text-white font-mono">{d.count}</span>
      </p>
      <p className="text-slate-500">{(d.pct * 100).toFixed(1)}% of runs</p>
    </div>
  );
};

export default function DistributionHistogram({
  bins = [],
  mean,
  var95,
  cvar95,
  color = '#3b82f6',
  height = 280,
}) {
  const total = bins.reduce((s, b) => s + b.count, 0) || 1;
  const data = bins.map((b) => ({ ...b, pct: b.count / total }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 10, right: 16, left: 6, bottom: 4 }} barCategoryGap={1}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="x"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: '#475569' }}
            tickLine={false}
            tickFormatter={(v) => `£${Math.round(v)}`}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={36}
          />
          <Tooltip content={<Tip />} cursor={{ fill: 'rgba(148,163,184,0.08)' }} />
          {typeof var95 === 'number' && (
            <ReferenceLine
              x={data.reduce((best, b) => (Math.abs(b.x - var95) < Math.abs(best - var95) ? b.x : best), data[0]?.x)}
              stroke="#f87171"
              strokeDasharray="4 2"
              label={{ value: 'VaR95', fill: '#f87171', fontSize: 10, position: 'top' }}
            />
          )}
          {typeof cvar95 === 'number' && (
            <ReferenceLine
              x={data.reduce((best, b) => (Math.abs(b.x - cvar95) < Math.abs(best - cvar95) ? b.x : best), data[0]?.x)}
              stroke="#dc2626"
              strokeDasharray="2 2"
              label={{ value: 'CVaR', fill: '#dc2626', fontSize: 10, position: 'insideTopLeft' }}
            />
          )}
          {typeof mean === 'number' && (
            <ReferenceLine
              x={data.reduce((best, b) => (Math.abs(b.x - mean) < Math.abs(best - mean) ? b.x : best), data[0]?.x)}
              stroke="#e2e8f0"
              strokeDasharray="4 2"
              label={{ value: 'Mean', fill: '#e2e8f0', fontSize: 10, position: 'top' }}
            />
          )}
          <Bar dataKey="count" radius={[2, 2, 0, 0]} isAnimationActive={false}>
            {data.map((b, i) => (
              <Cell key={i} fill={typeof var95 === 'number' && b.x <= var95 ? '#ef4444' : color} fillOpacity={typeof var95 === 'number' && b.x <= var95 ? 0.85 : 0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-5 mt-2 px-1 text-xs text-slate-400">
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ backgroundColor: color, opacity: 0.7 }} /> Outcomes</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-red-500" /> Tail (&le; VaR95)</span>
        <span className="flex items-center gap-1.5"><span className="w-4 h-0.5 bg-slate-200" /> Mean</span>
      </div>
    </div>
  );
}
