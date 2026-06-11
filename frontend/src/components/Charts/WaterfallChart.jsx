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
} from 'recharts';

// Transform waterfall data into stacked bar format with invisible baseline
function buildWaterfallData(rawData) {
  let running = 0;
  return rawData.map((item) => {
    if (item.type === 'total') {
      // Total bar starts from 0
      const base = 0;
      const bar = item.value;
      running = item.value;
      return { ...item, base, bar: Math.abs(bar), isNegative: bar < 0 };
    }
    // For flow items, stack on top of running
    const start = item.value >= 0 ? running : running + item.value;
    const height = Math.abs(item.value);
    running += item.value;
    return { ...item, base: start < 0 ? 0 : start, bar: height, isNegative: item.value < 0 };
  });
}

const BAR_COLORS = {
  positive: '#22c55e',
  negative: '#ef4444',
  total: '#3b82f6',
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const item = payload[0]?.payload;
  if (!item) return null;
  return (
    <div
      className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl"
      style={{ backgroundColor: '#0f172a' }}
    >
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      <p className="text-white font-mono font-bold">
        {item.value < 0 ? '-' : ''}£{Math.abs(item.value).toLocaleString()}M
      </p>
      <p className="text-slate-400 capitalize mt-0.5">{item.type}</p>
    </div>
  );
};

export default function WaterfallChart({ data = [] }) {
  const chartData = buildWaterfallData(data);

  const getColor = (entry) => {
    if (entry.type === 'total') return BAR_COLORS.total;
    if (entry.type === 'negative' || entry.isNegative) return BAR_COLORS.negative;
    return BAR_COLORS.positive;
  };

  return (
    <div
      className="rounded-xl p-4 border border-slate-700"
      style={{ backgroundColor: '#1e293b' }}
    >
      <div className="text-sm font-semibold text-slate-200 mb-4">
        P&L Waterfall (£M)
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: '#475569' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={60}
            tickFormatter={(v) => `£${v.toLocaleString()}`}
          />
          <Tooltip content={<CustomTooltip />} />
          {/* Invisible baseline bar */}
          <Bar dataKey="base" stackId="waterfall" fill="transparent" isAnimationActive={false} />
          {/* Visible value bar */}
          <Bar dataKey="bar" stackId="waterfall" radius={[3, 3, 0, 0]} isAnimationActive={true}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 px-2">
        {Object.entries(BAR_COLORS).map(([key, color]) => (
          <div key={key} className="flex items-center gap-1.5">
            <span
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: color }}
            />
            <span className="text-xs text-slate-400 capitalize">{key}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
