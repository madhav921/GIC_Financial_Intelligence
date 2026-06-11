import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import WaterfallChart from '../components/Charts/WaterfallChart';
import KPICard from '../components/Charts/KPICard';
import Badge from '../components/common/Badge';

const WATERFALL = [
  { label: 'Net Revenue',    value: 19800, cumulative: 19800, type: 'total' },
  { label: 'Material COGS',  value: -12700, cumulative: 7100,  type: 'negative' },
  { label: 'Gross Margin',   value: 7100,  cumulative: 7100,  type: 'total' },
  { label: 'Warranty',       value: -495,  cumulative: 6605,  type: 'negative' },
  { label: 'Depreciation',   value: -1140, cumulative: 5465,  type: 'negative' },
  { label: 'Other OpEx',     value: -4064, cumulative: 1401,  type: 'negative' },
  { label: 'EBIT',           value: 1401,  cumulative: 1401,  type: 'total' },
];

const MONTHLY_EBIT = Array.from({ length: 12 }, (_, i) => ({
  month: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i],
  ebit: Math.round((1401 / 12) * (0.85 + Math.random() * 0.3)),
}));

const SENSITIVITY = [
  { commodity: 'Steel',      bomWeight: '22%', impact1pct: -43.6, impact10pct: -436 },
  { commodity: 'Lithium',    bomWeight: '18%', impact1pct: -35.6, impact10pct: -356 },
  { commodity: 'Aluminum',   bomWeight: '12%', impact1pct: -23.8, impact10pct: -238 },
  { commodity: 'Cobalt',     bomWeight: '7%',  impact1pct: -13.9, impact10pct: -139 },
  { commodity: 'Copper',     bomWeight: '6%',  impact1pct: -11.9, impact10pct: -119 },
  { commodity: 'Nickel',     bomWeight: '5%',  impact1pct: -9.9,  impact10pct: -99  },
];

const SEGMENTS = [
  { segment: 'Luxury SUV',  revenue: 8400, volume: 80000,  margin: 22.1, cogs: 6540 },
  { segment: 'Premium SUV', revenue: 6640, volume: 120000, margin: 17.4, cogs: 5484 },
  { segment: 'Performance', revenue: 2960, volume: 65000,  margin: 15.8, cogs: 2493 },
  { segment: 'EV',          revenue: 1800, volume: 45000,  margin: 12.3, cogs: 1579 },
];

export default function FinancialPnL() {
  const [shockPct, setShockPct] = useState(0);

  const shockedEBIT = 1401 + SENSITIVITY.reduce((sum, s) => sum + s.impact10pct * (shockPct / 10), 0) / SENSITIVITY.length;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Financial P&amp;L</h1>
        <p className="text-slate-400 text-sm mt-1">Waterfall · Segment breakdown · Commodity cost sensitivity</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard title="Net Revenue"   value="£19.8B" subtitle="Annual" change="+4.2%" changeType="up" />
        <KPICard title="Gross Margin"  value="£7.1B"  subtitle="35.9%" change="-0.6pp" changeType="down" />
        <KPICard title="EBIT"          value="£1,401M" subtitle="7.1% margin" change="+8.3%" changeType="up" />
        <KPICard title="COGS / Revenue" value="64.1%"  subtitle="Material fraction 45%" change="+0.4pp" changeType="down" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Waterfall */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-1">P&amp;L Waterfall (£M)</h2>
          <p className="text-slate-500 text-xs mb-4">Revenue → COGS → Gross Margin → Fixed Costs → EBIT</p>
          <WaterfallChart data={WATERFALL} />
        </div>

        {/* Monthly EBIT */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-4">Monthly EBIT (£M)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={MONTHLY_EBIT} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
                formatter={(v) => [`£${v}M`, 'EBIT']}
              />
              <Bar dataKey="ebit" radius={[4,4,0,0]}>
                {MONTHLY_EBIT.map((entry, i) => (
                  <Cell key={i} fill={entry.ebit >= 120 ? '#22c55e' : entry.ebit >= 100 ? '#3b82f6' : '#f59e0b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Segment Table */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">Segment Analysis</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-slate-400 border-b border-slate-700">
              <th className="text-left pb-2">Segment</th>
              <th className="text-right pb-2">Revenue (£M)</th>
              <th className="text-right pb-2">COGS (£M)</th>
              <th className="text-right pb-2">Volume</th>
              <th className="text-right pb-2">Gross Margin %</th>
            </tr>
          </thead>
          <tbody>
            {SEGMENTS.map((s, i) => (
              <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/50">
                <td className="py-2 text-slate-200 font-medium">{s.segment}</td>
                <td className="py-2 text-right text-slate-200">£{s.revenue.toLocaleString()}M</td>
                <td className="py-2 text-right text-slate-300">£{s.cogs.toLocaleString()}M</td>
                <td className="py-2 text-right text-slate-300">{s.volume.toLocaleString()}</td>
                <td className={`py-2 text-right font-medium ${s.margin >= 20 ? 'text-green-400' : s.margin >= 15 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {s.margin}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Commodity Sensitivity */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-100">Commodity Cost Sensitivity</h2>
            <p className="text-slate-400 text-xs mt-1">EBIT impact of commodity price changes (BOM-weighted)</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-slate-400 text-sm">Shock:</span>
            <input
              type="range" min={-20} max={40} step={5} value={shockPct}
              onChange={e => setShockPct(Number(e.target.value))}
              className="w-32 accent-blue-500"
            />
            <span className={`font-bold text-sm w-14 text-right ${shockPct > 0 ? 'text-red-400' : shockPct < 0 ? 'text-green-400' : 'text-slate-300'}`}>
              {shockPct > 0 ? '+' : ''}{shockPct}%
            </span>
          </div>
        </div>

        {shockPct !== 0 && (
          <div className={`mb-4 rounded-lg px-4 py-3 border text-sm ${shockPct > 0 ? 'border-red-800 bg-red-900/20 text-red-300' : 'border-green-800 bg-green-900/20 text-green-300'}`}>
            Estimated EBIT impact of average {shockPct > 0 ? '+' : ''}{shockPct}% commodity shock:
            <span className="font-bold ml-2">£{Math.round(shockedEBIT - 1401)}M</span>
            {' '}({shockPct > 0 ? '▼' : '▲'} {Math.abs(((shockedEBIT / 1401) - 1) * 100).toFixed(1)}%)
          </div>
        )}

        <table className="w-full text-sm">
          <thead>
            <tr className="text-slate-400 border-b border-slate-700">
              <th className="text-left pb-2">Commodity</th>
              <th className="text-right pb-2">BOM Weight</th>
              <th className="text-right pb-2">EBIT impact (+1%)</th>
              <th className="text-right pb-2">EBIT impact (+10%)</th>
            </tr>
          </thead>
          <tbody>
            {SENSITIVITY.map((s, i) => (
              <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/50">
                <td className="py-2 text-slate-200 font-medium">{s.commodity}</td>
                <td className="py-2 text-right text-slate-400">{s.bomWeight}</td>
                <td className="py-2 text-right text-red-400">£{s.impact1pct}M</td>
                <td className="py-2 text-right text-red-400 font-medium">£{s.impact10pct}M</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
