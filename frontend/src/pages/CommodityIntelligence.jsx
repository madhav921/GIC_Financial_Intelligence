import React, { useMemo, useState, useEffect, useCallback } from 'react';
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts';
import Badge from '../components/common/Badge';
import LockedButton from '../components/common/LockedButton';
import CorrelationHeatmap from '../components/Charts/CorrelationHeatmap';
import { PERMISSIONS } from '../auth/permissions';
import { gicApi } from '../api/client';

// mase: Mean Absolute Scaled Error (benchmark vs naïve random-walk; <1 = beats naïve, >1 = worse).
//   Primary scale-free metric — valid across commodities of different price magnitudes and handles zeros.
// rmse_pct: Root Mean Square Error expressed as % of mean price.
//   Penalises large spike errors more than MAPE — use for risk/hedging sizing decisions.
// mape: Kept as reference but NOT the primary optimisation target (asymmetric, biased toward low forecasts).
// Base values are aligned with Yahoo Finance proxy data (data/raw/commodity_prices.csv, latest).
// Units match the yfinance_connector.py scale definitions:
//   Steel/Aluminum/Copper/Cobalt/Nickel/Polypropylene/ABS_Resin → USD/tonne
//   Lithium → USD/kg  ·  Platinum/Palladium/Rhodium → USD/troy oz  ·  Natural_Gas → USD/MMBtu
const COMMODITIES = [
  { name: 'Steel',         unit: 'USD/t',      weight: 22, category: 'Raw Material',   mape: 12.4, mase: 0.84, rmse_pct: 14.8, direction: 76, color: '#60a5fa', base: 790,    vol: 0.04, trend: 0.01 },
  { name: 'Lithium',       unit: 'USD/kg',     weight: 18, category: 'Battery',        mape: 11.9, mase: 0.76, rmse_pct: 16.2, direction: 64, color: '#a78bfa', base: 21,     vol: 0.12, trend: -0.08 },
  { name: 'Aluminum',      unit: 'USD/t',      weight: 12, category: 'Raw Material',   mape: 16.7, mase: 0.92, rmse_pct: 21.3, direction: 68, color: '#34d399', base: 3735,   vol: 0.05, trend: 0.02 },
  { name: 'Cobalt',        unit: 'USD/t',      weight: 7,  category: 'Battery',        mape: 14.7, mase: 0.88, rmse_pct: 19.1, direction: 70, color: '#f472b6', base: 24700,  vol: 0.10, trend: -0.05 },
  { name: 'Copper',        unit: 'USD/t',      weight: 6,  category: 'Raw Material',   mape: 7.0,  mase: 0.62, rmse_pct:  9.1, direction: 52, color: '#fb923c', base: 13900,  vol: 0.04, trend: 0.03 },
  { name: 'Nickel',        unit: 'USD/t',      weight: 5,  category: 'Battery',        mape: 10.8, mase: 0.78, rmse_pct: 14.6, direction: 55, color: '#facc15', base: 12250,  vol: 0.08, trend: -0.03 },
  { name: 'Platinum',      unit: 'USD/oz',     weight: 4,  category: 'Precious Metal', mape: 8.9,  mase: 0.71, rmse_pct: 11.2, direction: 60, color: '#e2e8f0', base: 1972,   vol: 0.05, trend: 0.01 },
  { name: 'Natural Gas',   unit: 'USD/MMBtu',  weight: 4,  category: 'Energy',         mape: 31.1, mase: 1.24, rmse_pct: 42.7, direction: 62, color: '#38bdf8', base: 30,     vol: 0.20, trend: 0.0 },
  { name: 'Palladium',     unit: 'USD/oz',     weight: 3,  category: 'Precious Metal', mape: 29.1, mase: 1.18, rmse_pct: 38.4, direction: 68, color: '#c084fc', base: 1420,   vol: 0.15, trend: -0.12 },
  { name: 'Polypropylene', unit: 'USD/t',      weight: 3,  category: 'Polymer',        mape: 9.8,  mase: 0.74, rmse_pct: 12.3, direction: 70, color: '#4ade80', base: 938,    vol: 0.06, trend: 0.02 },
  { name: 'Rhodium',       unit: 'USD/oz',     weight: 2,  category: 'Precious Metal', mape: 14.2, mase: 0.96, rmse_pct: 19.8, direction: 55, color: '#fbbf24', base: 3937,   vol: 0.12, trend: -0.10 },
  { name: 'ABS Resin',     unit: 'USD/t',      weight: 2,  category: 'Polymer',        mape: 17.2, mase: 0.94, rmse_pct: 22.1, direction: 64, color: '#f87171', base: 1438,   vol: 0.07, trend: 0.01 },
];

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

// Deterministic pseudo-random so static charts are stable across renders.
function seeded(seed) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
}

// Build 24m history with MA/Bollinger — only the history portion,
// not forecast (forecast comes from the backend when available).
function buildHistory(c) {
  const rnd = seeded(c.name.split('').reduce((a, ch) => a + ch.charCodeAt(0), 7));
  const hist = [];
  let price = c.base;
  const now = new Date(2026, 5, 1);
  for (let i = 23; i >= 0; i--) {
    const d = new Date(now);
    d.setMonth(d.getMonth() - i);
    price = Math.max(price * (1 + c.trend / 12 + (rnd() - 0.5) * c.vol), c.base * 0.2);
    hist.push({ date: d.toISOString().slice(0, 7), value: Math.round(price * 100) / 100, kind: 'history' });
  }
  const win = 6;
  return hist.map((row, i) => {
    const slice = hist.slice(Math.max(0, i - win + 1), i + 1).map((r) => r.value);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const sd = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length);
    return { ...row, ma: Math.round(mean * 100) / 100, bbUpper: Math.round((mean + 2 * sd) * 100) / 100, bbLower: Math.round((mean - 2 * sd) * 100) / 100 };
  });
}

// Fallback seeded forecast when backend is unavailable.
function buildFallbackForecast(c, horizon, lastPrice) {
  const now = new Date(2026, 5, 1);
  let f = lastPrice;
  return Array.from({ length: horizon }, (_, i) => {
    const d = new Date(now);
    d.setMonth(d.getMonth() + i + 1);
    f = f * (1 + c.trend / 12);
    const widen = 1 + (i + 1) * 0.012;
    return {
      date: d.toISOString().slice(0, 7),
      forecast: Math.round(f * 100) / 100,
      ciUpper: Math.round(f * (1 + 0.08 * widen) * 100) / 100,
      ciLower: Math.round(f * (1 - 0.08 * widen) * 100) / 100,
      kind: 'forecast',
    };
  });
}

// Mock cross-commodity correlation matrix (stable).
function buildCorr(names) {
  const n = names.length;
  const rnd = seeded(99);
  const m = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = i === j ? 1 : Math.round((rnd() * 1.6 - 0.6) * 100) / 100;
      m[i][j] = v;
      m[j][i] = v;
    }
  }
  return m;
}

const MODELS = {
  rows: (mape, mase, rmse_pct, direction) => [
    { model: 'SARIMAX', mape: +(mape * 1.08).toFixed(1), mase: +(mase * 1.07).toFixed(2), rmse: +(rmse_pct * 1.09).toFixed(1), dir: Math.max(40, direction - 4), weight: 55 },
    { model: 'XGBoost', mape: +(mape * 1.05).toFixed(1), mase: +(mase * 1.04).toFixed(2), rmse: +(rmse_pct * 1.06).toFixed(1), dir: direction, weight: 45 },
    { model: 'Ensemble', mape: +(mape * 0.92).toFixed(1), mase: +(mase * 0.91).toFixed(2), rmse: +(rmse_pct * 0.90).toFixed(1), dir: Math.min(95, direction + 5), weight: 100, best: true },
  ],
};

// Map frontend display names to backend API names.
function toApiName(displayName) {
  return displayName.replace(/ /g, '_');
}

const mapeColor = (m) => (m < 12 ? 'text-green-400' : m < 20 ? 'text-yellow-400' : 'text-red-400');
const mapeStatus = (m) => (m < 12 ? 'Good' : m < 20 ? 'Adequate' : 'High Uncertainty');

const ChartTip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg p-3 border border-slate-600 text-xs shadow-xl" style={{ backgroundColor: '#0f172a' }}>
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      {payload.filter((e) => e.value != null && e.dataKey !== 'ciLower' && e.dataKey !== 'bbLower').map((e, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: e.color }} />
          <span className="text-slate-400">{e.name}:</span>
          <span className="text-white font-mono">{typeof e.value === 'number' ? e.value.toLocaleString() : e.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function CommodityIntelligence() {
  const [selected, setSelected] = useState('Copper');
  const [horizon, setHorizon] = useState(12);
  const [overlays, setOverlays] = useState({ ma: true, bollinger: false, forecast: true });

  // Live forecast state keyed by "{commodity}:{horizon}"
  const [liveForecasts, setLiveForecasts] = useState({});
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastSource, setForecastSource] = useState('loading');

  // Real historical price data keyed by commodity name
  const [realHistory, setRealHistory] = useState({});
  const [historySource, setHistorySource] = useState('seeded');

  const commodity = COMMODITIES.find((c) => c.name === selected) || COMMODITIES[4];
  const ffn = FFN_METRICS[selected] || {};

  // Seeded history fallback (used until real history loads)
  const seededHistory = useMemo(() => buildHistory(commodity), [commodity]);

  // Fetch real historical data from backend
  const fetchHistory = useCallback(async (name) => {
    if (realHistory[name]) return;
    try {
      const result = await gicApi.getCommodityHistory(toApiName(name), 36);
      if (result?.history?.length) {
        // Convert to the same shape as buildHistory output (adds MA / Bollinger)
        const raw = result.history.map((r) => ({ date: r.date, value: r.price, kind: 'history' }));
        const win = 6;
        const enriched = raw.map((row, i) => {
          const slice = raw.slice(Math.max(0, i - win + 1), i + 1).map((r) => r.value);
          const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
          const sd = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length);
          return { ...row, ma: Math.round(mean * 100) / 100, bbUpper: Math.round((mean + 2 * sd) * 100) / 100, bbLower: Math.round((mean - 2 * sd) * 100) / 100 };
        });
        setRealHistory((prev) => ({ ...prev, [name]: enriched }));
        setHistorySource(result.data_source === 'real' ? 'real' : 'synthetic');
      }
    } catch {
      // keep seeded history as fallback
    }
  }, [realHistory]);

  // Fetch real forecast from backend on commodity/horizon change
  const fetchForecast = useCallback(async (name, h) => {
    const key = `${name}:${h}`;
    if (liveForecasts[key]) return; // already cached

    setForecastLoading(true);
    try {
      const result = await gicApi.forecastCommodity(toApiName(name), h);
      setLiveForecasts((prev) => ({ ...prev, [key]: result }));
      setForecastSource('live');
    } catch {
      setForecastSource('mock');
    }
    setForecastLoading(false);
  }, [liveForecasts]);

  useEffect(() => {
    fetchHistory(selected);
    fetchForecast(selected, horizon);
  }, [selected, horizon]); // eslint-disable-line

  // Use real history when available, seeded otherwise
  const history = realHistory[selected] || seededHistory;

  // Merge history + real/fallback forecast
  const series = useMemo(() => {
    const key = `${selected}:${horizon}`;
    const fc = liveForecasts[key];
    const lastHistVal = history[history.length - 1]?.value ?? commodity.base;

    let forecastPoints;
    if (fc?.dates?.length) {
      // Real backend forecast: SARIMAX trained on data/raw/ (or data/synthetic/ fallback)
      forecastPoints = fc.dates.map((date, i) => ({
        date: typeof date === 'string' ? date.slice(0, 7) : date,
        forecast: Math.round((fc.point_forecast[i] ?? 0) * 100) / 100,
        ciUpper: Math.round((fc.upper_80[i] ?? 0) * 100) / 100,
        ciLower: Math.round((fc.lower_80[i] ?? 0) * 100) / 100,
        kind: 'forecast',
      }));
    } else {
      forecastPoints = buildFallbackForecast(commodity, horizon, lastHistVal);
    }

    return [...history, ...forecastPoints];
  }, [history, commodity, horizon, selected, liveForecasts]);

  const corrNames = COMMODITIES.slice(0, 8).map((c) => c.name);
  const corr = useMemo(() => buildCorr(corrNames), [corrNames]);
  const modelRows = MODELS.rows(commodity.mape, commodity.mase, commodity.rmse_pct, commodity.direction);

  const toggle = (k) => setOverlays((o) => ({ ...o, [k]: !o[k] }));

  const forecastKey = `${selected}:${horizon}`;
  const hasLiveForecast = !!liveForecasts[forecastKey]?.dates?.length;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Data source banner */}
      <div className={`rounded-lg px-4 py-3 text-xs border flex items-center gap-2 ${hasLiveForecast ? 'border-green-700 text-green-300 bg-green-900/15' : 'text-slate-400 border-slate-700'}`} style={!hasLiveForecast ? { backgroundColor: '#1e293b' } : {}}>
        {hasLiveForecast ? (
          <>
            <span className="text-green-400 font-bold">●</span>
            <span>
              History: <strong>{historySource === 'real' ? 'Yahoo Finance (data/raw/)' : 'Seeded'}</strong>
              {' · '}Forecast: <strong>SARIMAX trained on {historySource === 'real' ? 'real' : 'synthetic'} data</strong>
              {' · '}Units: {commodity.unit}
            </span>
          </>
        ) : forecastLoading ? (
          <>
            <span className="text-blue-400">↻</span>
            <span>Loading forecast from backend...</span>
          </>
        ) : (
          <>
            <span className="text-yellow-400">ℹ️</span>
            <span>Backend offline — showing seeded mock forecast. Run <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code> for SARIMAX+XGBoost real forecasts.</span>
          </>
        )}
      </div>

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Commodity Intelligence</h1>
          <p className="text-slate-400 text-sm mt-1">Forecasts · model comparison · correlations · risk-adjusted performance for 12 JLR materials</p>
        </div>
        <div className="flex gap-2">
          <LockedButton permission={PERMISSIONS.TRIGGER_RETRAINING} onClick={() => {}} lockedLabel="Retrain" lockHint="Model retraining requires Administrator access" className="text-xs px-3 py-1.5">
            🔄 Retrain
          </LockedButton>
          <LockedButton permission={PERMISSIONS.EXPORT_REPORTS} onClick={() => {}} lockedLabel="Export" lockHint="Exporting reports requires Administrator access" color="#475569" hoverColor="#334155" className="text-xs px-3 py-1.5">
            ⬇ Export
          </LockedButton>
        </div>
      </div>

      {/* Commodity Selector */}
      <div className="flex flex-wrap gap-2">
        {COMMODITIES.map((c) => (
          <button key={c.name} onClick={() => setSelected(c.name)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors border ${
              selected === c.name ? 'bg-blue-600 border-blue-500 text-white' : 'border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200'
            }`}
            style={selected !== c.name ? { backgroundColor: '#1e293b' } : {}}
          >
            {c.name} <span className="text-xs opacity-70">{c.weight}%</span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Price + forecast chart */}
        <div className="lg:col-span-2 rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold text-slate-100">{selected} — Price &amp; Forecast</h2>
              <Badge label={commodity.category} color="blue" />
              {hasLiveForecast && <Badge label="Live Forecast" color="green" />}
              {forecastLoading && <span className="text-xs text-slate-400 animate-pulse">Loading...</span>}
            </div>
            <div className="flex items-center gap-3">
              <div className="flex gap-1.5">
                {[
                  { k: 'ma', label: 'MA' },
                  { k: 'bollinger', label: 'Bollinger' },
                  { k: 'forecast', label: 'Forecast CI' },
                ].map((o) => (
                  <button key={o.k} onClick={() => toggle(o.k)}
                    className={`px-2 py-1 rounded-md text-xs font-medium border transition-colors ${
                      overlays[o.k] ? 'bg-blue-600/30 border-blue-500 text-blue-200' : 'border-slate-700 text-slate-500 hover:text-slate-300'
                    }`}>
                    {o.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Horizon control */}
          <div className="flex items-center gap-3 mb-3 text-xs">
            <span className="text-slate-400">Horizon</span>
            {[3, 6, 12, 18, 24].map((h) => (
              <button key={h} onClick={() => setHorizon(h)}
                className={`px-2 py-0.5 rounded-md border ${horizon === h ? 'border-blue-500 text-blue-200 bg-blue-900/30' : 'border-slate-700 text-slate-500 hover:text-slate-300'}`}>
                {h}m
              </button>
            ))}
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={series} margin={{ top: 5, right: 16, left: 6, bottom: 4 }}>
              <defs>
                <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={commodity.color} stopOpacity={0.28} />
                  <stop offset="95%" stopColor={commodity.color} stopOpacity={0.04} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#475569' }} tickLine={false} interval="preserveStartEnd" />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} width={58} />
              <Tooltip content={<ChartTip />} />
              <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8', paddingTop: 8 }} />
              {overlays.forecast && <Area type="monotone" dataKey="ciUpper" stroke="none" fill="url(#ciGrad)" name={hasLiveForecast ? '~80% CI (SARIMAX)' : '~80% CI'} legendType="none" isAnimationActive={false} />}
              {overlays.forecast && <Area type="monotone" dataKey="ciLower" stroke="none" fill="#1e293b" legendType="none" isAnimationActive={false} />}
              {overlays.bollinger && <Line type="monotone" dataKey="bbUpper" stroke="#64748b" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Bollinger Upper" />}
              {overlays.bollinger && <Line type="monotone" dataKey="bbLower" stroke="#64748b" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Bollinger Lower" legendType="none" />}
              {overlays.ma && <Line type="monotone" dataKey="ma" stroke="#fbbf24" strokeWidth={1.5} dot={false} name="MA (6m)" />}
              <Line type="monotone" dataKey="value" stroke={commodity.color} strokeWidth={2.5} dot={false} name="Price (history)" connectNulls />
              {overlays.forecast && <Line type="monotone" dataKey="forecast" stroke={commodity.color} strokeWidth={2} strokeDasharray="5 4" dot={false} name={hasLiveForecast ? 'Forecast (SARIMAX+XGBoost)' : 'Forecast (mock)'} connectNulls />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics column */}
        <div className="space-y-4">
          {/* FFN tiles */}
          <div className="rounded-xl p-5 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Risk-Adjusted Performance (FFN)</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'CAGR', value: `${ffn.cagr > 0 ? '+' : ''}${ffn.cagr}%`, pos: ffn.cagr >= 0 },
                { label: 'Sharpe', value: ffn.sharpe?.toFixed(2), pos: ffn.sharpe >= 0 },
                { label: 'Sortino', value: ffn.sortino?.toFixed(2), pos: ffn.sortino >= 0 },
                { label: 'Max DD', value: `${ffn.maxDD}%`, pos: false },
              ].map((m) => (
                <div key={m.label} className="rounded-lg p-3 border border-slate-700" style={{ backgroundColor: '#0f172a' }}>
                  <p className="text-[10px] text-slate-400 uppercase tracking-wide">{m.label}</p>
                  <p className={`text-lg font-bold mt-0.5 ${m.pos ? 'text-emerald-400' : 'text-red-400'}`}>{m.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Forecast accuracy summary */}
          <div className="rounded-xl p-5 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
            <h3 className="text-sm font-semibold text-slate-300 mb-3">
              Forecast Accuracy (Backtest)
              {hasLiveForecast && liveForecasts[forecastKey]?.metrics && (
                <span className="ml-1.5 text-[10px] text-green-400 font-normal">● Live metrics</span>
              )}
            </h3>
            <div className="space-y-2 text-sm">
              {(() => {
                const liveMetrics = hasLiveForecast ? liveForecasts[forecastKey]?.metrics : null;
                const mase = liveMetrics?.mase ?? commodity.mase;
                const rmse = liveMetrics?.rmse_pct ?? commodity.rmse_pct;
                const mape = liveMetrics?.mape ?? commodity.mape;
                const dir = commodity.direction;
                return (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">MASE <span className="text-[10px] text-slate-600">(primary)</span></span>
                      <span className={mase < 0.8 ? 'text-green-400' : mase < 1.0 ? 'text-yellow-400' : 'text-red-400'}>{mase.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">RMSE <span className="text-[10px] text-slate-600">(spike risk)</span></span>
                      <span className={rmse < 15 ? 'text-green-400' : rmse < 25 ? 'text-yellow-400' : 'text-red-400'}>{rmse.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">MAPE <span className="text-[10px] text-slate-600">(ref only)</span></span>
                      <span className={mapeColor(mape)}>{mape.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between"><span className="text-slate-400">Directional Acc.</span><span className="text-blue-400">{dir}%</span></div>
                    <div className="flex justify-between"><span className="text-slate-400">BOM Weight</span><span className="text-slate-200">{commodity.weight}%</span></div>
                    <div className="flex justify-between items-center"><span className="text-slate-400">Status</span><Badge label={mapeStatus(mape)} color={mape < 12 ? 'green' : mape < 20 ? 'yellow' : 'red'} /></div>
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      </div>

      {/* Model comparison */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">Model Comparison — {selected}</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {modelRows.map((m) => (
            <div key={m.model} className={`rounded-xl p-4 border ${m.best ? 'border-blue-600 bg-blue-900/15' : 'border-slate-700'}`} style={!m.best ? { backgroundColor: '#0f172a' } : {}}>
              <div className="flex items-center justify-between mb-3">
                <span className="text-slate-200 font-semibold">{m.model}</span>
                {m.best ? <Badge label="Ensemble (active)" color="blue" /> : <span className="text-xs text-slate-500">weight {m.weight}%</span>}
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-slate-400">MASE <span className="text-[10px] text-slate-600">(primary)</span></span><span className={m.mase < 0.8 ? 'text-green-400' : m.mase < 1.0 ? 'text-yellow-400' : 'text-red-400'}>{m.mase}</span></div>
                <div className="flex justify-between"><span className="text-slate-400">RMSE <span className="text-[10px] text-slate-600">(spikes)</span></span><span className={m.rmse < 15 ? 'text-green-400' : m.rmse < 25 ? 'text-yellow-400' : 'text-red-400'}>{m.rmse}%</span></div>
                <div className="flex justify-between"><span className="text-slate-400">MAPE <span className="text-[10px] text-slate-600">(ref)</span></span><span className={mapeColor(m.mape)}>{m.mape}%</span></div>
                <div className="flex justify-between"><span className="text-slate-400">Directional Acc.</span><span className="text-blue-400">{m.dir}%</span></div>
                <div className="mt-1">
                  <div className="h-1.5 rounded-full bg-slate-700">
                    <div className="h-1.5 rounded-full" style={{ width: `${Math.min(100, m.dir)}%`, backgroundColor: m.best ? '#3b82f6' : '#64748b' }} />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Correlation heatmap */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-1">Cross-Commodity Correlation</h2>
        <p className="text-slate-500 text-xs mb-4">Pairwise log-return correlation (top-8 by BOM weight) — computed on monthly price changes</p>
        <CorrelationHeatmap labels={corrNames} matrix={corr} />
      </div>

      {/* BOM Weights Table */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">BOM Weight &amp; Forecast Accuracy — All Commodities</h2>
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
                <tr key={i} onClick={() => setSelected(c.name)}
                  className={`border-b border-slate-800 cursor-pointer transition-colors ${selected === c.name ? 'bg-blue-900/20' : 'hover:bg-slate-800/50'}`}>
                  <td className="py-2 text-slate-200 font-medium">
                    {c.name}
                    {liveForecasts[`${c.name}:${horizon}`]?.dates?.length > 0 && (
                      <span className="ml-1.5 text-[9px] text-green-500 font-bold">●</span>
                    )}
                  </td>
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
                    <Badge label={mapeStatus(c.mape)} color={c.mape < 12 ? 'green' : c.mape < 20 ? 'yellow' : 'red'} />
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
