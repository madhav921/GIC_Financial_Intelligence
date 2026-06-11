import React, { useState } from 'react';
import PriceChart from '../components/Charts/PriceChart';
import Badge from '../components/common/Badge';

const COMMODITIES = [
  { name: 'Steel',          weight: 22, category: 'Raw Material',   mape: 12.4, direction: 76, color: '#60a5fa' },
  { name: 'Lithium',        weight: 18, category: 'Battery',        mape: 11.9, direction: 64, color: '#a78bfa' },
  { name: 'Aluminum',       weight: 12, category: 'Raw Material',   mape: 16.7, direction: 68, color: '#34d399' },
  { name: 'Cobalt',         weight: 7,  category: 'Battery',        mape: 14.7, direction: 70, color: '#f472b6' },
  { name: 'Copper',         weight: 6,  category: 'Raw Material',   mape: 7.0,  direction: 52, color: '#fb923c' },
  { name: 'Nickel',         weight: 5,  category: 'Battery',        mape: 10.8, direction: 55, color: '#facc15' },
  { name: 'Platinum',       weight: 4,  category: 'Precious Metal', mape: 8.9,  direction: 60, color: '#e2e8f0' },
  { name: 'Natural Gas',    weight: 4,  category: 'Energy',         mape: 31.1, direction: 62, color: '#38bdf8' },
  { name: 'Palladium',      weight: 3,  category: 'Precious Metal', mape: 29.1, direction: 68, color: '#c084fc' },
  { name: 'Polypropylene',  weight: 3,  category: 'Polymer',        mape: 9.8,  direction: 70, color: '#4ade80' },
  { name: 'Rhodium',        weight: 2,  category: 'Precious Metal', mape: 14.2, direction: 55, color: '#fbbf24' },
  { name: 'ABS Resin',      weight: 2,  category: 'Polymer',        mape: 17.2, direction: 64, color: '#f87171' },
];

// Generate 24 months of fake price data
function mockPrices(basePrice, volatility, trend) {
  const months = [];
  let price = basePrice;
  const now = new Date(2026, 5, 1);
  for (let i = 23; i >= 0; i--) {
    const d = new Date(now);
    d.setMonth(d.getMonth() - i);
    price = price * (1 + trend / 12 + (Math.random() - 0.5) * volatility);
    price = Math.max(price, 1);
    months.push({
      date: d.toISOString().slice(0, 7),
      value: Math.round(price * 100) / 100,
      lower80: Math.round(price * 0.92 * 100) / 100,
      upper80: Math.round(price * 1.08 * 100) / 100,
    });
  }
  return months;
}

const PRICE_DATA = {
  Steel: mockPrices(650, 0.04, 0.03),
  Lithium: mockPrices(18000, 0.12, -0.08),
  Aluminum: mockPrices(2200, 0.05, 0.02),
  Cobalt: mockPrices(32000, 0.10, -0.05),
  Copper: mockPrices(8500, 0.04, 0.04),
  Nickel: mockPrices(16000, 0.08, -0.03),
  Platinum: mockPrices(980, 0.05, 0.01),
  'Natural Gas': mockPrices(3.2, 0.20, 0.0),
  Palladium: mockPrices(1050, 0.15, -0.12),
  Polypropylene: mockPrices(1100, 0.06, 0.02),
  Rhodium: mockPrices(4800, 0.12, -0.10),
  'ABS Resin': mockPrices(1400, 0.07, 0.01),
};

const FFN_METRICS = {
  Steel:       { cagr: 3.1, sharpe: 0.42, sortino: 0.58, maxDD: -18.4 },
  Lithium:     { cagr: -8.2, sharpe: -0.61, sortino: -0.82, maxDD: -65.3 },
  Aluminum:    { cagr: 2.0, sharpe: 0.31, sortino: 0.44, maxDD: -22.1 },
  Cobalt:      { cagr: -4.9, sharpe: -0.38, sortino: -0.52, maxDD: -48.7 },
  Copper:      { cagr: 4.3, sharpe: 0.55, sortino: 0.74, maxDD: -16.2 },
  Nickel:      { cagr: -2.8, sharpe: -0.22, sortino: -0.31, maxDD: -38.9 },
  Platinum:    { cagr: 1.1, sharpe: 0.14, sortino: 0.19, maxDD: -24.3 },
  'Natural Gas': { cagr: -0.4, sharpe: -0.03, sortino: -0.04, maxDD: -71.2 },
  Palladium:   { cagr: -12.3, sharpe: -0.89, sortino: -1.12, maxDD: -72.8 },
  Polypropylene: { cagr: 1.8, sharpe: 0.28, sortino: 0.39, maxDD: -20.1 },
  Rhodium:     { cagr: -9.7, sharpe: -0.71, sortino: -0.94, maxDD: -79.4 },
  'ABS Resin': { cagr: 1.2, sharpe: 0.17, sortino: 0.23, maxDD: -28.6 },
};

const mapeColor = (m) => m < 12 ? 'text-green-400' : m < 20 ? 'text-yellow-400' : 'text-red-400';
const mapeStatus = (m) => m < 12 ? 'Good' : m < 20 ? 'Adequate' : 'High Uncertainty';

export default function CommodityIntelligence() {
  const [selected, setSelected] = useState('Copper');
  const commodity = COMMODITIES.find(c => c.name === selected) || COMMODITIES[4];
  const priceData = PRICE_DATA[selected] || [];
  const ffn = FFN_METRICS[selected] || {};

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Commodity Intelligence</h1>
        <p className="text-slate-400 text-sm mt-1">Price forecasts · BOM weights · Performance analytics for 12 JLR materials</p>
      </div>

      {/* Commodity Selector */}
      <div className="flex flex-wrap gap-2">
        {COMMODITIES.map(c => (
          <button key={c.name}
            onClick={() => setSelected(c.name)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors border ${
              selected === c.name
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200'
            }`}
            style={selected !== c.name ? { backgroundColor: '#1e293b' } : {}}
          >
            {c.name} <span className="text-xs opacity-70">{c.weight}%</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Price Chart */}
        <div className="lg:col-span-2 rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-100">{selected} — 24-Month Price History</h2>
            <div className="flex gap-2">
              <Badge label={commodity.category} color="blue" />
              <Badge label={`BOM: ${commodity.weight}%`} color="green" />
            </div>
          </div>
          <PriceChart data={priceData} commodity={selected} color={commodity.color} />
        </div>

        {/* Metrics Panel */}
        <div className="space-y-4">
          {/* Forecast Accuracy */}
          <div className="rounded-xl p-5 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Forecast Accuracy (2024 Backtest)</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">CV MAPE</span>
                <span className={mapeColor(commodity.mape)}>{commodity.mape}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Directional Acc.</span>
                <span className="text-blue-400">{commodity.direction}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Status</span>
                <span className={mapeColor(commodity.mape)}>{mapeStatus(commodity.mape)}</span>
              </div>
            </div>
          </div>

          {/* FFN Analytics */}
          <div className="rounded-xl p-5 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">FFN Performance Metrics</h3>
            <div className="space-y-2">
              {[
                { label: 'CAGR', value: `${ffn.cagr > 0 ? '+' : ''}${ffn.cagr}%`, pos: ffn.cagr >= 0 },
                { label: 'Sharpe Ratio', value: ffn.sharpe?.toFixed(2), pos: ffn.sharpe >= 0 },
                { label: 'Sortino Ratio', value: ffn.sortino?.toFixed(2), pos: ffn.sortino >= 0 },
                { label: 'Max Drawdown', value: `${ffn.maxDD}%`, pos: false },
              ].map(m => (
                <div key={m.label} className="flex justify-between text-sm">
                  <span className="text-slate-400">{m.label}</span>
                  <span className={m.pos ? 'text-green-400' : 'text-red-400'}>{m.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* BOM Weights Table */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">BOM Weight & Forecast Accuracy — All Commodities</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left pb-2">Commodity</th>
                <th className="text-left pb-2">Category</th>
                <th className="text-right pb-2">BOM Weight</th>
                <th className="text-right pb-2">CV MAPE</th>
                <th className="text-right pb-2">Dir. Accuracy</th>
                <th className="text-right pb-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {COMMODITIES.map((c, i) => (
                <tr key={i}
                  onClick={() => setSelected(c.name)}
                  className={`border-b border-slate-800 cursor-pointer transition-colors ${
                    selected === c.name ? 'bg-blue-900/20' : 'hover:bg-slate-800/50'
                  }`}
                >
                  <td className="py-2 text-slate-200 font-medium">{c.name}</td>
                  <td className="py-2 text-slate-400">{c.category}</td>
                  <td className="py-2 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-16 h-1.5 rounded-full bg-slate-700">
                        <div className="h-1.5 rounded-full bg-blue-500" style={{ width: `${(c.weight / 22) * 100}%` }} />
                      </div>
                      <span className="text-slate-300 w-8 text-right">{c.weight}%</span>
                    </div>
                  </td>
                  <td className={`py-2 text-right ${mapeColor(c.mape)}`}>{c.mape}%</td>
                  <td className="py-2 text-right text-blue-400">{c.direction}%</td>
                  <td className="py-2 text-right">
                    <Badge
                      label={mapeStatus(c.mape)}
                      color={c.mape < 12 ? 'green' : c.mape < 20 ? 'yellow' : 'red'}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
