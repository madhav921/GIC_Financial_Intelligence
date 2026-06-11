import React, { useState } from 'react';
import Badge from '../components/common/Badge';

const DATASETS = [
  {
    name: 'market_commodities.parquet',
    type: 'real',
    rows: 2190,
    cols: 10,
    source: 'Yahoo Finance (yfinance)',
    updated: '2026-06-11',
    path: 'data/external/',
    description: '9 commodity price series + date. Daily bars resampled to monthly.',
    nullPct: 1.2,
    coverage: '2019-06 → 2026-06',
  },
  {
    name: 'fred_macro.parquet',
    type: 'real',
    rows: 732,
    cols: 12,
    source: 'FRED (Federal Reserve)',
    updated: '2026-06-11',
    path: 'data/external/',
    description: 'FEDFUNDS, CPI, PPI, INDPRO, UNRATE and 7 other macro series.',
    nullPct: 0.8,
    coverage: '2019-01 → 2026-06',
  },
  {
    name: 'market_indices.parquet',
    type: 'real',
    rows: 1825,
    cols: 8,
    source: 'Yahoo Finance',
    updated: '2026-06-11',
    path: 'data/external/',
    description: 'S&P 500, VIX, DJI, Gold, Oil, 10Y Treasury — daily.',
    nullPct: 0.0,
    coverage: '2021-01 → 2026-06',
  },
  {
    name: 'crypto_prices.parquet',
    type: 'real',
    rows: 1095,
    cols: 7,
    source: 'Binance via CCXT',
    updated: '2026-06-11',
    path: 'data/external/',
    description: 'BTC, ETH, SOL, XRP, BNB, AVAX — daily OHLCV.',
    nullPct: 0.0,
    coverage: '2022-01 → 2026-06',
  },
  {
    name: 'commodity_prices.csv',
    type: 'mixed',
    rows: 1560,
    cols: 13,
    source: 'Yahoo Finance + O-U Synthetic',
    updated: '2026-06-11',
    path: 'data/raw/',
    description: '12 commodities (9 real + 3 O-U synthetic: Rhodium, Polypropylene, ABS Resin). Monthly.',
    nullPct: 0.3,
    coverage: '2019-06 → 2026-06',
  },
  {
    name: 'macro_indicators.csv',
    type: 'mixed',
    rows: 1560,
    cols: 13,
    source: 'FRED + Synthetic',
    updated: '2026-06-11',
    path: 'data/raw/',
    description: '12 macro indicators: 8 real (FRED) + 4 synthetic (GDP Growth, China PPI, Baltic Dry, EV Sales).',
    nullPct: 2.1,
    coverage: '2019-01 → 2026-06',
  },
  {
    name: 'sales_data.csv',
    type: 'synthetic',
    rows: 480,
    cols: 8,
    source: 'JLR-calibrated Synthetic',
    updated: '2026-06-11',
    path: 'data/synthetic/',
    description: 'Vehicle sales by segment (Luxury SUV, Premium SUV, Performance, EV). Monthly.',
    nullPct: 0.0,
    coverage: '2019-01 → 2026-12',
  },
  {
    name: 'bom_data.csv',
    type: 'synthetic',
    rows: 12,
    cols: 4,
    source: 'Bill of Materials',
    updated: '2026-06-11',
    path: 'data/synthetic/',
    description: '12 commodity BOM weights (Steel 22%, Lithium 18%, Aluminum 12%, ...).',
    nullPct: 0.0,
    coverage: 'Static reference',
  },
  {
    name: 'production_inventory.csv',
    type: 'synthetic',
    rows: 480,
    cols: 6,
    source: 'JLR-calibrated Synthetic',
    updated: '2026-06-11',
    path: 'data/synthetic/',
    description: 'Monthly production volumes and inventory levels by segment.',
    nullPct: 0.0,
    coverage: '2019-01 → 2026-12',
  },
];

const typeColor = { real: 'green', mixed: 'yellow', synthetic: 'blue' };
const typeLabel = { real: 'Real', mixed: 'Real + Synthetic', synthetic: 'Synthetic' };

export default function DataExplorer() {
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState('all');

  const filtered = filter === 'all' ? DATASETS : DATASETS.filter(d => d.type === filter);
  const sel = DATASETS.find(d => d.name === selected);

  const totalRows = DATASETS.reduce((s, d) => s + d.rows, 0);
  const realCount = DATASETS.filter(d => d.type === 'real').length;
  const synthCount = DATASETS.filter(d => d.type === 'synthetic').length;

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Backend connect banner */}
      <div className="rounded-lg px-4 py-3 text-xs text-slate-400 border border-slate-700 flex items-center gap-2" style={{ backgroundColor: '#1e293b' }}>
        <span className="text-blue-400">ℹ️</span>
        Connect backend:{' '}
        <code className="text-blue-300 font-mono">uvicorn src.api.app:app --port 8000</code>
        {' '}· For real data run{' '}
        <code className="text-blue-300 font-mono">python scripts/fetch_data.py</code>
      </div>

      <div>
        <h1 className="text-2xl font-bold text-white">Data Explorer</h1>
        <p className="text-slate-400 text-sm mt-1">Dataset catalog · Data quality report · Source tracing</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Total Datasets', value: DATASETS.length, color: 'blue' },
          { label: 'Total Rows', value: totalRows.toLocaleString(), color: 'green' },
          { label: 'Real Data Sources', value: realCount, color: 'green' },
          { label: 'Synthetic Datasets', value: synthCount, color: 'yellow' },
        ].map((c, i) => (
          <div key={i} className="rounded-xl p-4 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
            <p className="text-slate-400 text-xs mb-1">{c.label}</p>
            <p className="text-2xl font-bold text-white">{c.value}</p>
          </div>
        ))}
      </div>

      {/* Filter */}
      <div className="flex gap-2">
        {['all', 'real', 'mixed', 'synthetic'].map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors capitalize border ${
              filter === f ? 'bg-blue-600 border-blue-500 text-white' : 'border-slate-700 text-slate-400 hover:border-slate-500'
            }`}
            style={filter !== f ? { backgroundColor: '#1e293b' } : {}}
          >
            {f === 'all' ? 'All Datasets' : typeLabel[f]}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Dataset List */}
        <div className="lg:col-span-2 rounded-xl border border-slate-700 overflow-hidden" style={{ backgroundColor: '#1e293b' }}>
          <div className="px-5 py-3 border-b border-slate-700">
            <h2 className="text-sm font-semibold text-slate-300">Datasets ({filtered.length})</h2>
          </div>
          <table className="w-full text-sm">
            <thead className="border-b border-slate-700">
              <tr className="text-slate-400 text-xs">
                <th className="text-left px-5 py-2">Name</th>
                <th className="text-right px-3 py-2">Rows</th>
                <th className="text-right px-3 py-2">Cols</th>
                <th className="text-right px-3 py-2">Null %</th>
                <th className="text-right px-5 py-2">Type</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d, i) => (
                <tr key={i}
                  onClick={() => setSelected(d.name === selected ? null : d.name)}
                  className={`border-b border-slate-800 cursor-pointer transition-colors ${
                    selected === d.name ? 'bg-blue-900/20' : 'hover:bg-slate-800/40'
                  }`}
                >
                  <td className="px-5 py-2.5">
                    <p className="text-slate-200 font-mono text-xs">{d.name}</p>
                    <p className="text-slate-500 text-xs mt-0.5">{d.path}</p>
                  </td>
                  <td className="px-3 py-2.5 text-right text-slate-300">{d.rows.toLocaleString()}</td>
                  <td className="px-3 py-2.5 text-right text-slate-300">{d.cols}</td>
                  <td className={`px-3 py-2.5 text-right ${d.nullPct > 5 ? 'text-red-400' : d.nullPct > 1 ? 'text-yellow-400' : 'text-green-400'}`}>
                    {d.nullPct}%
                  </td>
                  <td className="px-5 py-2.5 text-right">
                    <Badge label={typeLabel[d.type]} color={typeColor[d.type]} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Detail Panel */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          {sel ? (
            <>
              <h3 className="text-sm font-semibold text-slate-100 mb-1 font-mono">{sel.name}</h3>
              <Badge label={typeLabel[sel.type]} color={typeColor[sel.type]} />
              <div className="mt-4 space-y-3 text-sm">
                <div>
                  <p className="text-slate-400 text-xs">Description</p>
                  <p className="text-slate-200 mt-0.5">{sel.description}</p>
                </div>
                {[
                  { label: 'Source', value: sel.source },
                  { label: 'Path', value: sel.path, mono: true },
                  { label: 'Coverage', value: sel.coverage },
                  { label: 'Rows', value: sel.rows.toLocaleString() },
                  { label: 'Columns', value: sel.cols },
                  { label: 'Null %', value: `${sel.nullPct}%`, color: sel.nullPct > 5 ? 'text-red-400' : sel.nullPct > 1 ? 'text-yellow-400' : 'text-green-400' },
                  { label: 'Last Updated', value: sel.updated },
                ].map(r => (
                  <div key={r.label}>
                    <p className="text-slate-400 text-xs">{r.label}</p>
                    <p className={`mt-0.5 ${r.color || 'text-slate-200'} ${r.mono ? 'font-mono text-xs text-blue-400' : ''}`}>
                      {r.value}
                    </p>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-48 text-slate-500 text-sm text-center">
              <p className="text-2xl mb-3">🗄️</p>
              <p>Click a dataset row to see details</p>
            </div>
          )}
        </div>
      </div>

      {/* Scripts Reference */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-3">Data Scripts</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {[
            { cmd: 'python scripts/fetch_data.py',           desc: 'Fetch real data from Yahoo Finance, FRED, CCXT/Binance' },
            { cmd: 'python scripts/generate_data.py',        desc: 'Generate all synthetic datasets (no API key needed)' },
            { cmd: 'python scripts/train_models.py',         desc: 'Train SARIMAX + XGBoost for all 12 commodities' },
            { cmd: 'python scripts/run_commodity_pipeline.py', desc: 'Full 8-stage commodity forecast pipeline' },
          ].map((s, i) => (
            <div key={i} className="rounded-lg px-4 py-3 border border-slate-700" style={{ backgroundColor: '#0f172a' }}>
              <code className="text-blue-400 text-xs block mb-1">{s.cmd}</code>
              <p className="text-slate-400 text-xs">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
