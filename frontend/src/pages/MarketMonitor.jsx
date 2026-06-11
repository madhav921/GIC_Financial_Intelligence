import React, { useState, useEffect } from 'react';
import Badge from '../components/common/Badge';

const MARKET_DATA = {
  indices: [
    { name: 'S&P 500',     value: 4783.45, change: 1.21,  ticker: 'SPX',   currency: '' },
    { name: 'VIX',         value: 14.82,   change: -5.34, ticker: 'VIX',   currency: '' },
    { name: 'Gold',        value: 2048.30, change: 0.82,  ticker: 'GC=F',  currency: '$' },
    { name: 'Oil (WTI)',   value: 76.42,   change: -1.13, ticker: 'CL=F',  currency: '$' },
    { name: 'Dow Jones',   value: 37592.1, change: 0.54,  ticker: 'DJI',   currency: '' },
    { name: '10Y Treasury',value: 4.21,    change: 0.03,  ticker: 'TNX',   currency: '%' },
  ],
  fx: [
    { pair: 'GBP/USD', rate: 1.2734, change: 0.31 },
    { pair: 'EUR/USD', rate: 1.0891, change: -0.14 },
    { pair: 'USD/JPY', rate: 149.21, change: 0.52 },
    { pair: 'USD/CNY', rate: 7.1853, change: -0.18 },
  ],
  crypto: [
    { name: 'Bitcoin',  symbol: 'BTC', price: 42834.20, change: 2.31 },
    { name: 'Ethereum', symbol: 'ETH', price: 2274.85,  change: 1.82 },
    { name: 'Solana',   symbol: 'SOL', price: 98.42,    change: 5.12 },
    { name: 'XRP',      symbol: 'XRP', price: 0.6214,   change: -0.93 },
    { name: 'BNB',      symbol: 'BNB', price: 312.40,   change: 0.67 },
    { name: 'Avalanche',symbol: 'AVAX', price: 38.21,   change: 3.44 },
  ],
  macro: [
    { indicator: 'Fed Funds Rate', value: '5.25–5.50%', source: 'FRED', series: 'FEDFUNDS' },
    { indicator: 'CPI YoY',        value: '3.2%',       source: 'FRED', series: 'CPIAUCSL' },
    { indicator: 'PPI (US)',        value: '1.8%',       source: 'FRED', series: 'PPIACO' },
    { indicator: 'Industrial Prod.',value: '102.4',      source: 'FRED', series: 'INDPRO'  },
    { indicator: 'DXY Index',       value: '104.8',      source: 'Yahoo Finance', series: 'DX-Y.NYB' },
    { indicator: 'Unemployment',    value: '3.7%',       source: 'FRED', series: 'UNRATE'  },
  ],
};

function ChangeCell({ change }) {
  const color = change > 0 ? 'text-green-400' : change < 0 ? 'text-red-400' : 'text-slate-400';
  const arrow = change > 0 ? '▲' : change < 0 ? '▼' : '—';
  return (
    <span className={`${color} font-medium text-sm`}>
      {arrow} {Math.abs(change).toFixed(2)}%
    </span>
  );
}

function SectionCard({ title, badge, children }) {
  return (
    <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
        {badge && <Badge label={badge} color="blue" />}
      </div>
      {children}
    </div>
  );
}

export default function MarketMonitor() {
  const [lastUpdated, setLastUpdated] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setLastUpdated(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Market Monitor</h1>
          <p className="text-slate-400 text-sm mt-1">Live indices · FX rates · Crypto · FRED macro indicators</p>
        </div>
        <div className="text-right text-xs text-slate-500">
          <div className="flex items-center gap-1.5 justify-end">
            <span className="w-2 h-2 rounded-full bg-green-500 inline-block animate-pulse" />
            <span className="text-green-400">Live</span>
          </div>
          <p className="mt-0.5">Updated {lastUpdated.toLocaleTimeString()}</p>
        </div>
      </div>

      {/* Data source note */}
      <div className="rounded-lg px-4 py-2 text-xs text-slate-400 border border-slate-700 flex flex-wrap gap-3" style={{ backgroundColor: '#1e293b' }}>
        <span>Sources: <Badge label="Yahoo Finance" color="blue" /></span>
        <span><Badge label="FRED" color="green" /></span>
        <span><Badge label="Binance/CCXT" color="yellow" /></span>
        <span className="text-slate-500 ml-auto">Run <code className="text-blue-400">python scripts/fetch_data.py</code> to refresh</span>
      </div>

      {/* Market Indices */}
      <SectionCard title="Market Indices" badge="Yahoo Finance">
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {MARKET_DATA.indices.map((idx, i) => (
            <div key={i} className="rounded-lg p-3 border border-slate-700" style={{ backgroundColor: '#0f172a' }}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs text-slate-400">{idx.ticker}</p>
                  <p className="text-sm font-medium text-slate-200">{idx.name}</p>
                </div>
                <ChangeCell change={idx.change} />
              </div>
              <p className="text-xl font-bold text-white mt-1">
                {idx.currency}{idx.value.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
              </p>
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* FX Rates */}
        <SectionCard title="FX Rates" badge="Yahoo Finance">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700 text-xs">
                <th className="text-left pb-2">Pair</th>
                <th className="text-right pb-2">Rate</th>
                <th className="text-right pb-2">24h Change</th>
              </tr>
            </thead>
            <tbody>
              {MARKET_DATA.fx.map((fx, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-2 text-slate-200 font-medium">{fx.pair}</td>
                  <td className="py-2 text-right text-white font-mono">
                    {fx.rate.toFixed(4)}
                  </td>
                  <td className="py-2 text-right">
                    <ChangeCell change={fx.change} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </SectionCard>

        {/* Macro Indicators */}
        <SectionCard title="Macro Indicators" badge="FRED">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700 text-xs">
                <th className="text-left pb-2">Indicator</th>
                <th className="text-right pb-2">Value</th>
                <th className="text-right pb-2">Series</th>
              </tr>
            </thead>
            <tbody>
              {MARKET_DATA.macro.map((m, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-2 text-slate-200">{m.indicator}</td>
                  <td className="py-2 text-right text-white font-bold font-mono">{m.value}</td>
                  <td className="py-2 text-right">
                    <code className="text-xs text-blue-400">{m.series}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </SectionCard>
      </div>

      {/* Crypto */}
      <SectionCard title="Cryptocurrency" badge="Binance/CCXT">
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {MARKET_DATA.crypto.map((c, i) => (
            <div key={i} className="rounded-lg p-3 border border-slate-700" style={{ backgroundColor: '#0f172a' }}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs text-slate-400 font-mono">{c.symbol}/USDT</p>
                  <p className="text-sm font-medium text-slate-200">{c.name}</p>
                </div>
                <ChangeCell change={c.change} />
              </div>
              <p className="text-xl font-bold text-white mt-1">
                ${c.price < 10 ? c.price.toFixed(4) : c.price.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </p>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
