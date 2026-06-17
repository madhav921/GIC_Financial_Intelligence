import React, { useEffect, useState, useRef, useMemo } from 'react';
import Badge from '../components/common/Badge';
import Sparkline from '../components/Charts/Sparkline';
import LockedButton from '../components/common/LockedButton';
import { PERMISSIONS } from '../auth/permissions';
import { useRealtimeContext } from '../context/RealtimeContext';
import { gicApi } from '../api/client';

// Fallback index values shown when backend is unreachable.
const STATIC_INDICES = [
  { name: 'Gold',            ticker: 'GC=F',  currency: '$', value: 3250.0, change: 0.0 },
  { name: 'S&P 500',         ticker: '^GSPC', currency: '',  value: 5800.0, change: 0.0 },
  { name: 'VIX',             ticker: '^VIX',  currency: '',  value: 16.5,   change: 0.0 },
  { name: 'Oil (WTI)',       ticker: 'CL=F',  currency: '$', value: 72.0,   change: 0.0 },
  { name: '10Y Yield',       ticker: '^TNX',  currency: '%', value: 4.35,   change: 0.0 },
  { name: 'EURO STOXX Auto', ticker: 'SX7P',  currency: '',  value: 487.32, change: -1.24, isStatic: true },
];

const STATIC_MACRO = [
  { indicator: 'BoE Base Rate',       value: '4.25%',  source: 'BoE',      series: 'IUDSOIA' },
  { indicator: 'UK CPI YoY',          value: '3.5%',   source: 'ONS',      series: 'D7G7' },
  { indicator: 'UK PPI Output',        value: '1.8%',   source: 'ONS',      series: 'MM23' },
  { indicator: 'EU Industrial Prod.',  value: '99.2',   source: 'Eurostat', series: 'STS_INPR_M' },
  { indicator: 'DXY Index',            value: '99.8',   source: 'FRED',     series: 'DX-Y.NYB' },
  { indicator: 'UK Unemployment',      value: '4.5%',   source: 'ONS',      series: 'LFS' },
];

const FALLBACK_COMMODITIES = [
  { name: 'LME Aluminum', symbol: 'Al', price: 2490,  unit: '$/t',   change: 0.0, source: 'LME ref' },
  { name: 'LME Copper',   symbol: 'Cu', price: 9650,  unit: '$/t',   change: 0.0, source: 'LME ref' },
  { name: 'LME Steel HRC',symbol: 'St', price: 480,   unit: '$/t',   change: 0.0, source: 'LME ref' },
  { name: 'Lithium Carb.',symbol: 'Li', price: 10200, unit: '$/t',   change: 0.0, source: 'Fastmarkets ref' },
  { name: 'Cobalt',       symbol: 'Co', price: 24500, unit: '$/t',   change: 0.0, source: 'LME ref' },
  { name: 'TTF Gas',      symbol: 'Gas',price: 35.0,  unit: '€/MWh', change: 0.0, source: 'ICE ref' },
];

// Extract numeric price array from various commodity history response shapes.
function extractPrices(data, apiName) {
  if (!data) return [];
  const hist = data.history;
  if (!hist) return [];
  if (Array.isArray(hist)) {
    return hist.map(h => Number(h.price ?? h.value ?? h[apiName] ?? 0)).filter(v => v > 0);
  }
  if (typeof hist === 'object') {
    return Object.values(hist).map(Number).filter(v => v > 0);
  }
  return [];
}

// Deterministic LCG seed → 28-point buffer scaled around anchorValue.
// vol is a fraction of anchorValue (e.g. 0.0012 = 0.12% per step).
function makeSeedBuffer(seed, anchorValue, vol, points = 28) {
  let s = (Math.abs(seed * 16807 + 1337) % 2147483647) || 1;
  const rnd = () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
  const arr = [];
  let v = anchorValue;
  for (let i = 0; i < points; i++) {
    v += (rnd() - 0.5) * Math.abs(anchorValue) * vol * 3;
    arr.push(v);
  }
  return arr;
}

/**
 * Maintains a rolling 28-point buffer that ticks forward every `intervalMs` ms
 * using a slow Ornstein-Uhlenbeck mean-reverting walk.
 * Re-seeds itself if the anchor changes by >3% (e.g. fallback → real price loads).
 */
function useLiveSparkline(anchorValue, {
  vol = 0.0012,
  reversion = 0.015,
  intervalMs = 4000,
  points = 28,
  seed = 0,
} = {}) {
  const [buffer, setBuffer] = useState(() => makeSeedBuffer(seed, anchorValue, vol, points));

  const anchorRef = useRef(anchorValue);
  anchorRef.current = anchorValue;

  const prevAnchorRef = useRef(anchorValue);

  // If the anchor jumps >3% (e.g. real data arrives), re-seed so the sparkline
  // doesn't look wrong while mean-reverting across a large gap.
  useEffect(() => {
    const prev = prevAnchorRef.current;
    if (prev === 0) { prevAnchorRef.current = anchorValue; return; }
    const pct = Math.abs(anchorValue - prev) / Math.abs(prev);
    if (pct > 0.03) {
      prevAnchorRef.current = anchorValue;
      setBuffer(makeSeedBuffer(seed, anchorValue, vol, points));
    }
  }, [anchorValue, seed, vol, points]);

  // Tick the buffer forward every intervalMs.
  useEffect(() => {
    const id = setInterval(() => {
      const anchor = anchorRef.current;
      setBuffer(prev => {
        const last = prev[prev.length - 1];
        const drift = reversion * (anchor - last);
        const noise = (Math.random() - 0.5) * Math.abs(anchor) * vol * 2;
        const next = last + drift + noise;
        return [...prev.slice(1), next];
      });
    }, intervalMs);
    return () => clearInterval(id);
  }, [vol, reversion, intervalMs]);

  return buffer;
}

// CommodityTile — live price from WebSocket, animated sparkline, real MoM % when history loaded.
// Falls back to per-tick change_pct from the WebSocket when history hasn't arrived yet.
function CommodityTile({ c, tileIndex }) {
  const seed = c.name.split('').reduce((acc, ch, i) => acc * 31 + ch.charCodeAt(0) + i, tileIndex + 200);
  const liveBuffer = useLiveSparkline(c.price, {
    vol: 0.0012,
    reversion: 0.015,
    intervalMs: 2000,
    points: 28,
    seed,
  });

  const sparkData = c.sparkData.length >= 2 ? c.sparkData : liveBuffer;
  const last2 = sparkData.slice(-2);
  const isUp = last2.length < 2 ? true : last2[1] >= last2[0];

  return (
    <div className="rounded-lg p-3 border border-slate-700 transition-colors hover:border-slate-500" style={{ backgroundColor: '#0f172a' }}>
      <div className="flex justify-between items-start">
        <div>
          <p className="text-xs text-slate-400 font-mono">
            {c.symbol} · {c.source}
            {c.isLive && <span className="ml-1 text-green-500 font-bold">●</span>}
          </p>
          <p className="text-sm font-medium text-slate-200">{c.name}</p>
        </div>
        <ChangeCell change={c.change} label={c.changeLabel} />
      </div>
      <div className="flex items-end justify-between mt-1">
        <p className="text-xl font-bold text-white">
          {c.unit?.startsWith('$') ? '$' : c.unit?.startsWith('£') ? '£' : c.unit?.startsWith('€') ? '€' : ''}
          {c.price.toLocaleString('en-GB', { minimumFractionDigits: c.price < 100 ? 2 : 0 })}
          <span className="text-xs text-slate-500 ml-1 font-normal">
            {c.unit?.replace(/^[$£€]/, '')}
          </span>
        </p>
        <Sparkline data={sparkData} color={isUp ? '#22c55e' : '#ef4444'} width={88} height={26} />
      </div>
    </div>
  );
}

// ChangeCell — shows arrow + % + small interval label for commodity tiles.
function ChangeCell({ change, label }) {
  const color = change > 0 ? 'text-green-400' : change < 0 ? 'text-red-400' : 'text-slate-400';
  const arrow = change > 0 ? '▲' : change < 0 ? '▼' : '—';
  return (
    <span className={`${color} font-medium text-sm whitespace-nowrap`}>
      {arrow} {Math.abs(change).toFixed(2)}%
      {label && <span className="text-slate-500 text-[10px] font-normal ml-1">{label}</span>}
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

// Market index tile — animated sparkline (seeded, slow O-U walk, updates ~4s).
// Real 1d change from Yahoo Finance shown below the value as a reference.
function IndexTile({ idx, tileIndex }) {
  const seed = idx.ticker.split('').reduce((acc, c, i) => acc * 31 + c.charCodeAt(0) + i, tileIndex + 1);
  const sparkData = useLiveSparkline(idx.value, {
    vol: 0.0012,
    reversion: 0.015,
    intervalMs: 4000,
    points: 28,
    seed,
  });
  const isUp = sparkData.length >= 2
    ? sparkData[sparkData.length - 1] >= sparkData[sparkData.length - 2]
    : true;

  return (
    <div className="rounded-lg p-3 border border-slate-700 transition-colors hover:border-slate-500" style={{ backgroundColor: '#0f172a' }}>
      <div className="mb-1">
        <p className="text-xs text-slate-400 font-mono">
          {idx.ticker}
          {idx.isLive && <span className="ml-1.5 text-[9px] text-green-500 font-bold">●</span>}
          {idx.isStatic && <span className="ml-1.5 text-[9px] text-yellow-500">ref</span>}
        </p>
        <p className="text-sm font-medium text-slate-200">{idx.name}</p>
      </div>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-xl font-bold text-white">
            {idx.currency !== '%' && idx.currency}
            {idx.value.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
            {idx.currency === '%' && '%'}
          </p>
          {/* Real 1d change from Yahoo Finance — shown for reference only */}
          <p className="text-[10px] mt-0.5 flex items-center gap-1">
            <span className="text-slate-500">1d</span>
            {idx.isLive ? (
              <span className={idx.change > 0 ? 'text-green-400' : idx.change < 0 ? 'text-red-400' : 'text-slate-400'}>
                {idx.change > 0 ? '+' : ''}{idx.change.toFixed(2)}%
              </span>
            ) : idx.isStatic ? (
              <span className={idx.change > 0 ? 'text-green-400' : idx.change < 0 ? 'text-red-400' : 'text-slate-400'}>
                {idx.change > 0 ? '+' : ''}{idx.change.toFixed(2)}%
                <span className="text-slate-600 ml-1">ref</span>
              </span>
            ) : (
              <span className="text-slate-600">— ref</span>
            )}
            {idx.isLive && <span className="text-slate-600">· YF</span>}
          </p>
        </div>
        <Sparkline data={sparkData} color={isUp ? '#22c55e' : '#ef4444'} width={88} height={28} />
      </div>
      {idx.asOf && idx.asOf !== 'reference' && (
        <p className="text-[9px] text-slate-600 mt-0.5">Close: {idx.asOf}</p>
      )}
    </div>
  );
}

// FX table row — animated sparkline + real 1d % shown below the rate.
function FxRow({ fx, rowIndex }) {
  const seed = fx.pair.split('').reduce((acc, c, i) => acc * 31 + c.charCodeAt(0) + i, rowIndex + 100);
  const sparkData = useLiveSparkline(fx.rate, {
    vol: 0.0006,
    reversion: 0.02,
    intervalMs: 4000,
    points: 28,
    seed,
  });
  const isUp = fx.change >= 0;

  return (
    <tr className="border-b border-slate-800">
      <td className="py-2 text-slate-200 font-medium text-sm">
        {fx.pair}
        {fx.isLive && <span className="ml-1.5 text-[9px] text-green-500 font-bold">●</span>}
      </td>
      <td className="py-2 text-right">
        <div className="text-white font-mono text-sm">{fx.rate.toFixed(4)}</div>
        {/* Real 1d change from Yahoo Finance shown below rate as reference */}
        {fx.changeLabel !== 'ref' ? (
          <div className="text-[9px] text-slate-500">
            1d{' '}
            <span className={fx.change > 0 ? 'text-green-400' : fx.change < 0 ? 'text-red-400' : 'text-slate-400'}>
              {fx.change > 0 ? '+' : ''}{fx.change.toFixed(3)}%
            </span>
            <span className="text-slate-600 ml-1">· YF</span>
          </div>
        ) : (
          <div className="text-[9px] text-slate-600">ref</div>
        )}
      </td>
      <td className="py-2">
        <div className="flex justify-center">
          <Sparkline data={sparkData} color={isUp ? '#22c55e' : '#ef4444'} width={88} height={26} />
        </div>
      </td>
    </tr>
  );
}

export default function MarketMonitor() {
  const { snapshot, connected, source } = useRealtimeContext();
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [liveIndices, setLiveIndices] = useState(null);
  const [indicesSource, setIndicesSource] = useState('loading');
  const [indicesNote, setIndicesNote] = useState('');

  // Real monthly history per commodity (fetched once for MoM sparklines).
  const [commodityHistories, setCommodityHistories] = useState({});
  // Real 30-day FX history from Yahoo Finance (1d change reference only).
  const [fxHistoryData, setFxHistoryData] = useState(null);
  const histFetched = useRef(false);

  useEffect(() => {
    if (snapshot) setLastUpdated(new Date());
  }, [snapshot]);

  // Fetch real commodity monthly history once when first snapshot arrives.
  useEffect(() => {
    if (!snapshot?.top_commodities?.length || histFetched.current) return;
    histFetched.current = true;
    for (const c of snapshot.top_commodities) {
      const apiName = c.name.replace(/ /g, '_');
      gicApi.getCommodityHistory(apiName, 24)
        .then(data => {
          const prices = extractPrices(data, apiName);
          if (prices.length) {
            setCommodityHistories(prev => ({ ...prev, [c.name]: prices }));
          }
        })
        .catch(() => {});
    }
  }, [snapshot?.top_commodities]);

  // On mount: fetch live indices and real FX 1d reference data.
  useEffect(() => {
    gicApi.getMarketIndices()
      .then(data => {
        setLiveIndices(data.indices || []);
        setIndicesSource(data.data_source || 'Reference');
        setIndicesNote(data.note || '');
      })
      .catch(() => setIndicesSource('Reference (backend offline)'));

    gicApi.getFxHistory()
      .then(data => setFxHistoryData(data))
      .catch(() => {});
  }, []);

  // Merge live API data into STATIC_INDICES (keyed by ticker).
  const displayIndices = useMemo(() => {
    const liveMap = {};
    if (liveIndices) {
      for (const idx of liveIndices) liveMap[idx.ticker] = idx;
    }
    return STATIC_INDICES.map(s => {
      const live = liveMap[s.ticker];
      if (live) {
        return { ...s, value: live.value, change: live.change_pct, asOf: live.as_of, isLive: true };
      }
      return { ...s, isLive: false };
    });
  }, [liveIndices]);

  const indicesIsLive = indicesSource.includes('Yahoo Finance');
  const indicesBadge = indicesSource === 'loading' ? 'Loading…' : indicesIsLive ? 'Live' : 'Reference';
  const indicesBadgeColor = indicesIsLive ? 'green' : 'yellow';

  const backendDataSource = snapshot?.data_source || (source === 'live' ? 'Yahoo Finance' : 'Simulated');

  // Commodity tiles: live price (ticks every 2s) + real MoM change when history loaded,
  // otherwise falls back to per-tick change_pct from the WebSocket snapshot.
  const commodityTiles = useMemo(() => {
    if (snapshot?.top_commodities?.length) {
      return snapshot.top_commodities.map(c => {
        const hist = commodityHistories[c.name] || [];
        let momChange = parseFloat((c.change_pct || 0).toFixed(2));
        let changeLabel = '2s';
        if (hist.length >= 2) {
          const last = hist[hist.length - 1];
          const prev = hist[hist.length - 2];
          momChange = prev > 0 ? parseFloat(((last - prev) / prev * 100).toFixed(2)) : momChange;
          changeLabel = 'MoM';
        }
        return {
          name: c.name,
          symbol: c.name.slice(0, 2).toUpperCase(),
          price: c.price,
          unit: c.unit || 'USD/t',
          change: momChange,
          changeLabel,
          sparkData: hist,
          source: backendDataSource,
          isLive: true,
        };
      });
    }
    return FALLBACK_COMMODITIES.map(c => ({ ...c, changeLabel: 'ref', isLive: false, sparkData: [] }));
  }, [snapshot?.top_commodities, backendDataSource, commodityHistories]);

  // FX rows: live rate from WebSocket + real 1d % reference from Yahoo Finance.
  const fxRows = useMemo(() => {
    const histPairs = fxHistoryData?.pairs || {};
    const liveFx = {};
    if (snapshot?.fx?.length) {
      for (const f of snapshot.fx) liveFx[f.pair] = f.rate;
    }

    const makeRow = (pair, fallbackRate) => {
      const hData = histPairs[pair];
      const rate = liveFx[pair] ?? hData?.history?.[hData.history.length - 1]?.rate ?? fallbackRate;
      const change1d = hData?.change_1d_pct ?? 0;
      const changeLabel = hData ? '1d' : 'ref';
      const isLive = !!(liveFx[pair] || hData);
      return { pair, rate, change: change1d, changeLabel, isLive };
    };

    return [
      makeRow('GBP/USD', 1.3426),
      { pair: 'GBP/EUR', rate: 1.1563, change: 0, changeLabel: 'ref', isLive: false },
      makeRow('EUR/USD', 1.1612),
      { pair: 'USD/JPY', rate: 147.20, change: 0, changeLabel: 'ref', isLive: false },
      makeRow('USD/CNY', 6.7557),
    ];
  }, [snapshot?.fx, fxHistoryData]);

  const fxSource = fxHistoryData?.data_source || (source === 'live' ? 'Yahoo Finance' : 'Reference');
  const fxIsLive = fxSource.includes('Yahoo Finance');
  const dataSourceLabel = source === 'live' ? 'Live' : 'Simulated';
  const dataSourceColor = source === 'live' ? 'green' : 'yellow';

  const handleRefresh = () => {
    setIndicesSource('loading');
    gicApi.refreshMarketData().catch(() => {});
    gicApi.getMarketIndices()
      .then(data => {
        setLiveIndices(data.indices || []);
        setIndicesSource(data.data_source || 'Reference');
        setIndicesNote(data.note || '');
      })
      .catch(() => setIndicesSource('Reference'));
    gicApi.getFxHistory()
      .then(data => setFxHistoryData(data))
      .catch(() => {});
    setLastUpdated(new Date());
  };

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
            onClick={handleRefresh}
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
        <span>Commodity price feed:</span>
        <Badge label={dataSourceLabel} color={dataSourceColor} />
        {source === 'live' ? (
          <>
            <span className="text-green-400">
              WebSocket connected · Prices from <strong>{backendDataSource}</strong>, mean-reverting every 2s
            </span>
            <span className="text-slate-500">
              · ETF/futures proxies (SLX→Steel, LIT→Lithium, HG=F→Copper) · Commodity change% = real MoM
            </span>
          </>
        ) : (
          <span className="text-yellow-400">
            Backend offline — client-side simulator active. Run{' '}
            <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code> for live data.
          </span>
        )}
        <span className="text-slate-600 ml-auto">
          Indices & FX charts: animated simulation · 1d % = real Yahoo Finance · Macro: FRED / ONS / BoE
        </span>
      </div>

      {/* Market Indices — animated sparklines, real 1d % below value */}
      <SectionCard title="Market Indices" badge={indicesBadge} badgeColor={indicesBadgeColor}>
        <p className="text-xs text-slate-500 mb-3">
          {indicesIsLive
            ? (indicesNote || 'Prev close · Yahoo Finance') + ' · Chart animates live · 1d % shown below each value'
            : indicesSource === 'loading'
              ? 'Fetching from Yahoo Finance…'
              : 'Reference values · Start backend for live prices · Charts animate for visual liveliness'}
        </p>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {displayIndices.map((idx, i) => (
            <IndexTile key={idx.ticker} idx={idx} tileIndex={i} />
          ))}
        </div>
      </SectionCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* FX Rates — animated sparklines, real 1d % from Yahoo Finance below each rate */}
        <SectionCard
          title="FX Rates"
          badge={fxIsLive ? 'Yahoo Finance' : dataSourceLabel}
          badgeColor={fxIsLive ? 'green' : 'yellow'}
        >
          <p className="text-xs text-slate-500 mb-3">
            Rate ticks live every 2s · Chart animates · 1d % shown below rate (Yahoo Finance)
          </p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700 text-xs">
                <th className="text-left pb-2">Pair</th>
                <th className="text-right pb-2">Rate</th>
                <th className="text-center pb-2">Live chart</th>
              </tr>
            </thead>
            <tbody>
              {fxRows.map((fx, i) => (
                <FxRow key={fx.pair} fx={fx} rowIndex={i} />
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
                    <Badge
                      label={m.source}
                      color={['FRED', 'BoE', 'ONS'].includes(m.source) ? 'green' : 'blue'}
                    />
                  </td>
                  <td className="py-2 text-right">
                    <code className="text-xs text-blue-400">{m.series}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </SectionCard>
      </div>

      {/* Automotive Commodity Spot Prices — live price, real MoM change, real monthly sparkline */}
      <SectionCard title="Automotive Commodity Spot Prices" badge={dataSourceLabel} badgeColor={dataSourceColor}>
        <p className="text-xs text-slate-500 mb-3">
          {source === 'live'
            ? `${backendDataSource} · ETF/futures proxies · Price ticks live every 2s · Change = real MoM from Yahoo Finance · Sparkline = real monthly history`
            : 'Simulated prices — run backend for Yahoo Finance live data'}
          {' · '}Key supply-chain inputs against GIC's £3.3B commodity basket
        </p>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {commodityTiles.map((c, i) => (
            <CommodityTile key={i} c={c} tileIndex={i} />
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
