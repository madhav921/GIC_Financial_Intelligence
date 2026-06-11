import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl"
      style={{ backgroundColor: '#0f172a' }}
    >
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      {payload.map((entry, i) => {
        if (!entry.value && entry.value !== 0) return null;
        return (
          <div key={i} className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-slate-400">{entry.name}:</span>
            <span className="text-white font-mono">
              {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
            </span>
          </div>
        );
      })}
    </div>
  );
};

export default function PriceChart({ data = [], commodity = 'Commodity', color = '#3b82f6' }) {
  const hasCI = data.some((d) => d.lower80 !== undefined || d.upper80 !== undefined);

  return (
    <div
      className="rounded-xl p-4 border border-slate-700"
      style={{ backgroundColor: '#1e293b' }}
    >
      <div className="text-sm font-semibold text-slate-200 mb-4">
        {commodity} — Price Trend & Forecast
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id={`ciGrad-${commodity}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.25} />
              <stop offset="95%" stopColor={color} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: '#475569' }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={55}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '12px', color: '#94a3b8', paddingTop: '8px' }}
          />
          {hasCI && (
            <Area
              type="monotone"
              dataKey="upper80"
              stroke="none"
              fill={`url(#ciGrad-${commodity})`}
              name="80% CI Upper"
              legendType="none"
            />
          )}
          {hasCI && (
            <Area
              type="monotone"
              dataKey="lower80"
              stroke="none"
              fill="#1e293b"
              name="80% CI Lower"
              legendType="none"
            />
          )}
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 4, fill: color }}
            name="Price"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
