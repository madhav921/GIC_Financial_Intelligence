import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload || {};
  return (
    <div
      className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl"
      style={{ backgroundColor: '#0f172a' }}
    >
      <p className="text-slate-300 font-medium mb-2">{label}</p>
      <div className="space-y-1">
        {data.p95 !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">P95:</span>
            <span className="text-white font-mono">£{Number(data.p95).toFixed(0)}M</span>
          </div>
        )}
        {data.p90 !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">P90:</span>
            <span className="text-white font-mono">£{Number(data.p90).toFixed(0)}M</span>
          </div>
        )}
        {data.p75 !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">P75:</span>
            <span className="text-white font-mono">£{Number(data.p75).toFixed(0)}M</span>
          </div>
        )}
        {data.mean !== undefined && (
          <div className="flex justify-between gap-4 border-t border-slate-700 pt-1 mt-1">
            <span className="text-blue-400 font-semibold">Mean:</span>
            <span className="text-blue-300 font-mono font-bold">£{Number(data.mean).toFixed(0)}M</span>
          </div>
        )}
        {data.p25 !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">P25:</span>
            <span className="text-white font-mono">£{Number(data.p25).toFixed(0)}M</span>
          </div>
        )}
        {data.p10 !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">P10:</span>
            <span className="text-white font-mono">£{Number(data.p10).toFixed(0)}M</span>
          </div>
        )}
        {data.p5 !== undefined && (
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">P5:</span>
            <span className="text-white font-mono">£{Number(data.p5).toFixed(0)}M</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default function FanChart({ data = [], title = 'Monte Carlo P&L Fan Chart', color = '#3b82f6' }) {
  return (
    <div
      className="rounded-xl p-4 border border-slate-700"
      style={{ backgroundColor: '#1e293b' }}
    >
      <div className="text-sm font-semibold text-slate-200 mb-4">{title}</div>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id="fanGrad90" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.08} />
              <stop offset="95%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="fanGrad75" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.18} />
              <stop offset="95%" stopColor={color} stopOpacity={0.06} />
            </linearGradient>
            <linearGradient id="fanGrad50" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.32} />
              <stop offset="95%" stopColor={color} stopOpacity={0.12} />
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
            width={65}
            tickFormatter={(v) => `£${v}M`}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* 90% band (outermost) — p5 to p95 */}
          <Area
            type="monotone"
            dataKey="p95"
            stroke="none"
            fill="url(#fanGrad90)"
            name="90th pct"
            legendType="none"
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="p5"
            stroke="none"
            fill="#1e293b"
            name="5th pct"
            legendType="none"
            isAnimationActive={false}
          />

          {/* 75% band — p25 to p75 */}
          <Area
            type="monotone"
            dataKey="p75"
            stroke="none"
            fill="url(#fanGrad75)"
            name="75th pct"
            legendType="none"
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="p25"
            stroke="none"
            fill="#1e293b"
            name="25th pct"
            legendType="none"
            isAnimationActive={false}
          />

          {/* Mean line */}
          <Area
            type="monotone"
            dataKey="mean"
            stroke={color}
            strokeWidth={2.5}
            fill="none"
            dot={false}
            activeDot={{ r: 4, fill: color }}
            name="Mean EBIT"
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-5 mt-3 px-2">
        <div className="flex items-center gap-1.5">
          <span className="w-6 h-3 rounded-sm" style={{ backgroundColor: color, opacity: 0.3 }} />
          <span className="text-xs text-slate-400">90% CI</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-6 h-3 rounded-sm" style={{ backgroundColor: color, opacity: 0.55 }} />
          <span className="text-xs text-slate-400">50% CI</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-6 h-0.5" style={{ backgroundColor: color }} />
          <span className="text-xs text-slate-400">Mean</span>
        </div>
      </div>
    </div>
  );
}
