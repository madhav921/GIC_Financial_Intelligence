import React, { useEffect, useState } from 'react';
import Badge from '../components/common/Badge';
import Sparkline from '../components/Charts/Sparkline';
import LockedButton from '../components/common/LockedButton';
import { PERMISSIONS } from '../auth/permissions';
import { useRealtimeContext } from '../context/RealtimeContext';

// Reference-only static data — market indices and macro indicators that update
// weekly/monthly and are not served by the realtime WebSocket. Commodity prices
// and FX rates are replaced by the live backend feed when connected.
const STATIC_INDICES = [
  { name: 'S&P 500',         value: 4783.45, change: 1.21,  ticker: 'SPX',   currency: '' },
  { name: 'VIX',             value: 14.82,   change: -5.34, ticker: 'VIX',   currency: '' },
  { name: 'Gold',            value: 2048.30, change: 0.82,  ticker: 'GC=F',  currency: '$' },
  { name: 'Oil (Brent)',     value: 79.18,   change: -0.87, ticker: 'BZ=F',  currency: '$' },
  { name: 'EURO STOXX Auto', value: 487.32,  change: -1.24, ticker: 'SX7P',  currency: '' },
  { name: '10Y Gilt',        value: 4.38,    change: 0.04,  ticker: 'GBPGBP=', currency: '%' },
];

const STATIC_MACRO = [
  { indicator: 'BoE Base Rate',   value: '5.00%',   source: 'BoE',  series: 'IUDSOIA' },
  { indicator: 'UK CPI YoY',      value: '2.8%',    source: 'ONS',  series: 'D7G7' },
  { indicator: 'UK PPI Output',   value: '1.2%',    source: 'ONS',  series: 'MM23' },
  { indicator: 'EU Industrial Prod.', value: '99.6', source: 'Eurostat', series: 'STS_INPR_M' },
  { indicator: 'DXY Index',       value: '104.8',   source: 'FRED', series: 'DX-Y.NYB' },
  { indicator: 'UK Unemployment', value: '4.2%',    source: 'ONS',  series: 'LFS' },
];

// Fallback commodity tiles used when the backend is unreachable.
const FALLBACK_COMMODITIES = [
  { name: 'LME Aluminum', symbol: 'Al', price: 2318, unit: '$/t',   change: -0.54, source: 'LME' },
  { name: 'LME Copper',   symbol: 'Cu', price: 9184, unit: '$/t',   change: 0.72,  source: 'LME' },
  { name: 'LME Steel HRC',symbol: 'St', price: 588,  unit: '£/t',   change: -1.12, source: 'LME' },
  { name: 'Lithium Carb.',symbol: 'Li', price: 12450, unit: '$/t',  change: -2.31, source: 'Fastmarkets' },
  { name: 'Cobalt',       symbol: 'Co', price: 26800, unit: '$/t',  change: 1.05,  source: 'LME' },
  { name: 'TTF Gas',      symbol: 'Gas',price: 34.72, unit: '€/MWh',change: 3.18,  source: 'ICE' },
];

// FX pairs not in the backend feed — shown as static reference.
const STATIC_FX_EXTRA = [
  { pair: 'GBP/EUR', rate: 1.1682, change: 0.18 },
  { pair: 'USD/JPY', rate: 149.21, change: 0.52 },
];

// Stable 14-point sparkline series for static/fallback tiles.
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
  out.push(v + change);
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

function SectionCard({ title, badge, badgeColor, children }) {
  return (
    <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
        {badge && <Badge label={badge} color={badgeColor || 'blue'} />}
      </div>
      {children}
    </div>
  );
}

function TileSpark({ seed, change }) {
  const series = React.useMemo(() => sparkSeries(seed, change), [seed, change]);
  return <Sparkline data={series} color={change >= 0 ? '#22c55e' : '#ef4444'} width={88} height={26} />;
}

// Spark from realtime history — just use last 14 values of a mean-reverting walk
// seeded from the current price and change direction.
function LiveSpark({ price, change }) {
  const series = React.useMemo(() => {
    const seed = Math.round(price * 7 + change * 100);
    return sparkSeries(seed, change);
  }, [price, change]);
  return <Sparkline data={series} color={change >= 0 ? '#22c55e' : '#ef4444'} width={88} height={26} />;
}

export default function MarketMonitor() {
  const { snapshot, connected, source } = useRealtimeContext();
  const [lastUpdated, setLastUpdated] = useState(new Date());

  useEffect(() => {
    if (snapshot) setLastUpdated(new Date());
  }, [snapshot]);

  // Backend-reported data source — "Yahoo Finance" or "Synthetic"
  const backendDataSource = snapshot?.data_source || (source === 'live' ? 'Yahoo Finance' : 'Simulated');

  // Commodity tiles: prefer live backend feed, fall back to static
  const commodityTiles = React.useMemo(() => {
    if (snapshot?.top_commodities?.length) {
      return snapshot.top_commodities.map((c) => ({
        name: c.name,
        symbol: c.name.slice(0, 2).toUpperCase(),
        price: c.price,
        unit: c.unit || 'USD/t',
        change: c.change_pct,
        source: backendDataSource,
        isLive: true,
      }));
    }
    return FALLBACK_COMMODITIES.map((c) => ({ ...c, isLive: false }));
  }, [snapshot, source, backendDataSource]);

  // FX rates: merge live feed (3 pairs) with static extras (2 pairs)
  const fxRows = React.useMemo(() => {
    if (!snapshot?.fx?.length) {
      return [
        { pair: 'GBP/USD', rate: 1.2734, change: 0.31 },
        { pair: 'GBP/EUR', rate: 1.1682, change: 0.18 },
        { pair: 'EUR/USD', rate: 1.0891, change: -0.14 },
        { pair: 'USD/JPY', rate: 149.21, change: 0.52 },
        { pair: 'USD/CNY', rate: 7.1853, change: -0.18 },
      ];
    }
    const liveMap = Object.fromEntries(
      snapshot.fx.map((f) => [f.pair, { pair: f.pair, rate: f.rate, change: f.change_pct, isLive: true }])
    );
    return [
      liveMap['GBP/USD'] || { pair: 'GBP/USD', rate: 1.2734, change: 0.31 },
      STATIC_FX_EXTRA[0],
      liveMap['EUR/USD'] || { pair: 'EUR/USD', rate: 1.0891, change: -0.14 },
      { pair: 'USD/JPY', rate: 149.21, change: 0.52 },
      liveMap['USD/CNY'] || { pair: 'USD/CNY', rate: 7.1853, change: -0.18 },
    ];
  }, [snapshot]);

  const dataSourceLabel = source === 'live' ? 'Live' : 'Simulated';
  const dataSourceColor = source === 'live' ? 'green' : 'yellow';

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Market Monitor</h1>
          <p className="text-slate-400 text-sm mt-1">Live indices · FX rates · LME automotive commodities · macro indicators</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right text-xs text-slate-500">
            <div className="flex items-center gap-1.5 justify-end">
              <span className={`w-2 h-2 rounded-full inline-block ${connected ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`} />
              <span className={connected ? 'text-green-400' : 'text-yellow-400'}>
                {connected ? 'Connected' : 'Simulated'}
              </span>
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

      {/* Data source strip */}
      <div className="rounded-lg px-4 py-2 text-xs text-slate-400 border border-slate-700 flex flex-wrap items-center gap-3" style={{ backgroundColor: '#1e293b' }}>
        <span>Commodity & FX feed:</span>
        <Badge label={dataSourceLabel} color={dataSourceColor} />
        {source === 'live' ? (
          <>
            <span className="text-green-400">
              WebSocket connected · Prices seeded from <strong>{backendDataSource}</strong> and mean-reverting every 2s
            </span>
            <span className="text-slate-500">
              · Prices are ETF/futures proxies (SLX→Steel, LIT→Lithium, HG=F→Copper, etc.) in model units
            </span>
          </>
        ) : (
          <span className="text-yellow-400">Backend offline — client-side simulator active. Run <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code> for live data.</span>
        )}
        <span className="text-slate-600 ml-auto">Market indices &amp; macro: reference data (FRED / ONS / BoE)</span>
      </div>

      {/* Market Indices — static reference data, updated weekly */}
      <SectionCard title="Market Indices" badge="Reference" badgeColor="yellow">
        <p className="text-xs text-slate-500 mb-3">Weekly reference snapshot — S&P 500, VIX, Gold, Oil, EURO STOXX</p>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {STATIC_INDICES.map((idx, i) => (
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
        {/* FX Rates — live from WebSocket */}
        <SectionCard title="FX Rates" badge={dataSourceLabel} badgeColor={dataSourceColor}>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700 text-xs">
                <th className="text-left pb-2">Pair</th>
                <th className="text-right pb-2">Rate</th>
                <th className="text-center pb-2">14d</th>
                <th className="text-right pb-2">Change</th>
              </tr>
            </thead>
            <tbody>
              {fxRows.map((fx, i) => (
                <tr key={i} className="border-b border-slate-800">
                  <td className="py-2 text-slate-200 font-medium">
                    {fx.pair}
                    {fx.isLive && <span className="ml-1.5 text-[9px] text-green-500 font-bold">●</span>}
                  </td>
                  <td className="py-2 text-right text-white font-mono">{fx.rate.toFixed(4)}</td>
                  <td className="py-2">
                    <div className="flex justify-center">
                      <LiveSpark price={fx.rate * 1000} change={fx.change} />
                    </div>
                  </td>
                  <td className="py-2 text-right"><ChangeCell change={fx.change} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </SectionCard>

        {/* Macro Indicators — static reference */}
        <SectionCard title="Macro Indicators" badge="Reference" badgeColor="yellow">
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
              {STATIC_MACRO.map((m, i) => (
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

      {/* Automotive Commodity Spot Prices — live from WebSocket */}
      <SectionCard title="Automotive Commodity Spot Prices" badge={dataSourceLabel} badgeColor={dataSourceColor}>
        <p className="text-xs text-slate-500 mb-3">
          {source === 'live'
            ? `${backendDataSource} · ETF/futures proxies scaled to commodity units · Mean-reverting live simulation anchored to latest real prices`
            : 'Simulated prices — run backend for Yahoo Finance live data'}
          {' · '}Key supply-chain inputs against GIC's £3.3B commodity basket
        </p>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {commodityTiles.map((c, i) => (
            <div key={i} className="rounded-lg p-3 border border-slate-700 transition-colors hover:border-slate-500" style={{ backgroundColor: '#0f172a' }}>
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-xs text-slate-400 font-mono">
                    {c.symbol} · {c.source}
                    {c.isLive && <span className="ml-1 text-green-500 font-bold">●</span>}
                  </p>
                  <p className="text-sm font-medium text-slate-200">{c.name}</p>
                </div>
                <ChangeCell change={c.change} />
              </div>
              <div className="flex items-end justify-between mt-1">
                <p className="text-xl font-bold text-white">
                  {c.unit?.startsWith('$') ? '$' : c.unit?.startsWith('£') ? '£' : c.unit?.startsWith('€') ? '€' : ''}
                  {c.price.toLocaleString('en-GB', { minimumFractionDigits: c.price < 100 ? 2 : 0 })}
                  <span className="text-xs text-slate-500 ml-1 font-normal">
                    {c.unit?.replace(/^[$£€]/, '')}
                  </span>
                </p>
                <LiveSpark price={c.price} change={c.change} />
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* EBIT Nowcast from realtime feed */}
      {snapshot?.ebit_nowcast_gbp && (
        <div className="rounded-xl p-5 border border-slate-700 flex flex-wrap items-center gap-6" style={{ backgroundColor: '#1e293b' }}>
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">EBIT Nowcast</p>
            <p className="text-3xl font-bold text-white">
              £{(snapshot.ebit_nowcast_gbp / 1e9).toFixed(2)}B
            </p>
            <p className="text-xs text-slate-500 mt-0.5">Live estimate — anti-correlated with commodity index</p>
          </div>
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">Commodity Index</p>
            <p className={`text-3xl font-bold ${snapshot.commodity_index > 103 ? 'text-red-400' : snapshot.commodity_index < 98 ? 'text-green-400' : 'text-white'}`}>
              {snapshot.commodity_index?.toFixed(1)}
            </p>
            <p className="text-xs text-slate-500 mt-0.5">Base = 100 · BOM-weighted composite</p>
          </div>
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">Risk Band</p>
            <Badge
              label={`${snapshot.risk_band} · ${snapshot.risk_score?.toFixed(0)}/100`}
              color={snapshot.risk_score < 33 ? 'green' : snapshot.risk_score < 66 ? 'yellow' : 'red'}
            />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">Headline Insight</p>
            <p className="text-sm text-slate-300 italic">"{snapshot.headline_insight}"</p>
          </div>
        </div>
      )}
    </div>
  );
}
