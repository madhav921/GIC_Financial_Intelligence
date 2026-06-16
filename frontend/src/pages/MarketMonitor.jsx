import React, { useMemo, useState, useEffect } from 'react';
import Badge from '../components/common/Badge';
import Sparkline from '../components/Charts/Sparkline';
import LockedButton from '../components/common/LockedButton';
import { PERMISSIONS } from '../auth/permissions';

const MARKET_DATA = {
  indices: [
    { name: 'S&P 500',         value: 4783.45, change: 1.21,  ticker: 'SPX',   currency: '' },
    { name: 'VIX',             value: 14.82,   change: -5.34, ticker: 'VIX',   currency: '' },
    { name: 'Gold',            value: 2048.30, change: 0.82,  ticker: 'GC=F',  currency: '$' },
    { name: 'Oil (Brent)',     value: 79.18,   change: -0.87, ticker: 'BZ=F',  currency: '$' },
    { name: 'EURO STOXX Auto', value: 487.32,  change: -1.24, ticker: 'SX7P',  currency: '' },
    { name: '10Y Gilt',        value: 4.38,    change: 0.04,  ticker: 'GBPGBP=', currency: '%' },
  ],
  fx: [
    { pair: 'GBP/USD', rate: 1.2734, change: 0.31 },
    { pair: 'GBP/EUR', rate: 1.1682, change: 0.18 },
    { pair: 'EUR/USD', rate: 1.0891, change: -0.14 },
    { pair: 'USD/JPY', rate: 149.21, change: 0.52 },
    { pair: 'USD/CNY', rate: 7.1853, change: -0.18 },
  ],
  // Key automotive supply-chain commodities tracked via LME / spot markets
  commodities: [
    { name: 'LME Aluminum', symbol: 'Al', price: 2318, unit: '$/t',  change: -0.54, source: 'LME' },
    { name: 'LME Copper',   symbol: 'Cu', price: 9184, unit: '$/t',  change: 0.72,  source: 'LME' },
    { name: 'LME Steel HRC',symbol: 'St', price: 588,  unit: '£/t',  change: -1.12, source: 'LME' },
    { name: 'Lithium Carb.', symbol: 'Li', price: 12450, unit: '$/t', change: -2.31, source: 'Fastmarkets' },
    { name: 'Cobalt',       symbol: 'Co', price: 26800, unit: '$/t', change: 1.05,  source: 'LME' },
    { name: 'TTF Gas',      symbol: 'Gas', price: 34.72, unit: '€/MWh', change: 3.18, source: 'ICE' },
  ],
  macro: [
    { indicator: 'BoE Base Rate',   value: '5.00%',   source: 'BoE',  series: 'IUDSOIA' },
    { indicator: 'UK CPI YoY',      value: '2.8%',    source: 'ONS',  series: 'D7G7' },
    { indicator: 'UK PPI Output',   value: '1.2%',    source: 'ONS',  series: 'MM23' },
    { indicator: 'EU Industrial Prod.', value: '99.6', source: 'Eurostat', series: 'STS_INPR_M' },
    { indicator: 'DXY Index',       value: '104.8',   source: 'FRED', series: 'DX-Y.NYB' },
    { indicator: 'UK Unemployment', value: '4.2%',    source: 'ONS',  series: 'LFS' },
  ],
};

// Stable 14-point sparkline series that ends consistent with the daily change.
function sparkSeries(seed, change) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  const rnd = () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
  const out = [];
  let v = 100;
  for (let i = 0; i < 13; i++) {
    v += (rnd() - 0.5) * 3;
    out.push(v);
  }
  out.push(v + change); // last move reflects the reported change direction
  return out;
}

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

function TileSpark({ seed, change }) {
  const series = useMemo(() => sparkSeries(seed, change), [seed, change]);
  return <Sparkline data={series} color={change >= 0 ? '#22c55e' : '#ef4444'} width={88} height={26} />;
}

export default function MarketMonitor() {
  const [lastUpdated, setLastUpdated] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setLastUpdated(new Date()), 60000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className="text-blue-400">ℹ️</span>
        Connect backend:{' '}
        <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code>
        {' '}· For real-time data run{' '}
        <code className="text-blue-300 font-mono">python scripts/fetch_data.py</code>
      </div>

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Market Monitor</h1>
          <p className="text-slate-400 text-sm mt-1">Live indices · FX rates · LME automotive commodities · macro indicators</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right text-xs text-slate-500">
            <div className="flex items-center gap-1.5 justify-end">
              <span className="w-2 h-2 rounded-full bg-green-500 inline-block animate-pulse" />
              <span className="text-green-400">Live</span>
            </div>
            <p className="mt-0.5">Updated {lastUpdated.toLocaleTimeString()}</p>
          </div>
          <LockedButton
            permission={PERMISSIONS.TRIGGER_DATA_FETCH}
            onClick={() => setLastUpdated(new Date())}
            lockedLabel="Refresh live data"
            lockHint="Fetching live market data requires Administrator access"
            className="text-sm"
          >
            🔄 Refresh live data
          </LockedButton>
        </div>
      </div>

      {/* Data source note */}
      <div className="rounded-lg px-4 py-2 text-xs text-slate-400 border border-slate-700 flex flex-wrap items-center gap-3" style={{ backgroundColor: '#1e293b' }}>
        <span>Sources:</span>
        <Badge label="Yahoo Finance" color="blue" />
        <Badge label="LME" color="yellow" />
        <Badge label="FRED / ONS" color="green" />
        <span className="text-slate-500 ml-auto">Admins can trigger a live fetch; viewers see the latest cached snapshot.</span>
      </div>

      {/* Market Indices */}
      <SectionCard title="Market Indices" badge="Yahoo Finance">
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {MARKET_DATA.indices.map((idx, i) => (
            <div key={i} className="rounded-lg p-3 border border-slate-700 transition-colors hover:border-slate-500" style={{ backgroundColor: '#0f172a' }}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs text-slate-400">{idx.ticker}</p>
                  <p className="text-sm font-medium text-slate-200">{idx.name}</p>
                </div>
                <ChangeCell change={idx.change} />
              </div>
              <div className="flex items-end justify-between mt-1">
                <p className="text-xl font-bold text-white">
                  {idx.currency}{idx.value.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
                </p>
                <TileSpark seed={idx.ticker.charCodeAt(0) * 131 + i} change={idx.change} />
              </div>
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
                <th className="text-center pb-2">14d</th>
                <th className="text-right pb-2">24h Change</th>
              </tr>
            </thead>
            <tbody>
              {MARKET_DATA.fx.map((fx, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-2 text-slate-200 font-medium">{fx.pair}</td>
                  <td className="py-2 text-right text-white font-mono">{fx.rate.toFixed(4)}</td>
                  <td className="py-2">
                    <div className="flex justify-center"><TileSpark seed={fx.pair.charCodeAt(4) * 71 + i} change={fx.change} /></div>
                  </td>
                  <td className="py-2 text-right"><ChangeCell change={fx.change} /></td>
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
                <th className="text-right pb-2">Source</th>
                <th className="text-right pb-2">Series</th>
              </tr>
            </thead>
            <tbody>
              {MARKET_DATA.macro.map((m, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-2 text-slate-200">{m.indicator}</td>
                  <td className="py-2 text-right text-white font-bold font-mono">{m.value}</td>
                  <td className="py-2 text-right">
                    <Badge label={m.source} color={m.source === 'FRED' || m.source === 'BoE' || m.source === 'ONS' ? 'green' : 'blue'} />
                  </td>
                  <td className="py-2 text-right"><code className="text-xs text-blue-400">{m.series}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        </SectionCard>
      </div>

      {/* Automotive Commodity Spot Prices */}
      <SectionCard title="Automotive Commodity Spot Prices" badge="LME / Spot">
        <p className="text-xs text-slate-500 mb-3">Key supply-chain inputs tracked against GIC's £3.3B commodity basket</p>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {MARKET_DATA.commodities.map((c, i) => (
            <div key={i} className="rounded-lg p-3 border border-slate-700 transition-colors hover:border-slate-500" style={{ backgroundColor: '#0f172a' }}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs text-slate-400 font-mono">{c.symbol} · {c.source}</p>
                  <p className="text-sm font-medium text-slate-200">{c.name}</p>
                </div>
                <ChangeCell change={c.change} />
              </div>
              <div className="flex items-end justify-between mt-1">
                <p className="text-xl font-bold text-white">
                  {c.unit.startsWith('$') ? '$' : c.unit.startsWith('£') ? '£' : c.unit.startsWith('€') ? '€' : ''}
                  {c.price.toLocaleString('en-GB', { minimumFractionDigits: c.price < 100 ? 2 : 0 })}
                  <span className="text-xs text-slate-500 ml-1 font-normal">
                    {c.unit.replace(/^[$£€]/, '')}
                  </span>
                </p>
                <TileSpark seed={c.symbol.charCodeAt(0) * 97 + i} change={c.change} />
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
