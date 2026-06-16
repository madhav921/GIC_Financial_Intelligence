import React, { useState } from 'react';
import Badge from '../components/common/Badge';
import { useAuth } from '../auth/AuthContext';
import { can, PERMISSIONS } from '../auth/permissions';

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
    source: 'FRED / ONS / BoE',
    updated: '2026-06-11',
    path: 'data/external/',
    description: 'BoE base rate, UK CPI, UK PPI, EU industrial production, DXY and 7 other macro series.',
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
    description: 'S&P 500, VIX, Gold, Brent Oil, EURO STOXX Auto (SX7P), 10Y Gilt — daily.',
    nullPct: 0.0,
    coverage: '2021-01 → 2026-06',
  },
  {
    name: 'lme_commodities.parquet',
    type: 'real',
    rows: 912,
    cols: 7,
    source: 'LME / Fastmarkets / ICE',
    updated: '2026-06-11',
    path: 'data/external/',
    description: 'LME Aluminum, Copper, Steel HRC; Lithium Carbonate spot; Cobalt; TTF Natural Gas — monthly.',
    nullPct: 0.4,
    coverage: '2019-01 → 2026-06',
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

// Lightweight per-dataset schema + aggregated stats (always visible) and a
// small raw-row preview that is gated behind view_raw_data (Admin).
const SCHEMAS = {
  'market_commodities.parquet': [
    { col: 'date', dtype: 'datetime64', nulls: 0.0 },
    { col: 'copper', dtype: 'float64', nulls: 1.1 },
    { col: 'steel', dtype: 'float64', nulls: 1.4 },
    { col: 'aluminum', dtype: 'float64', nulls: 0.9 },
  ],
  'fred_macro.parquet': [
    { col: 'date', dtype: 'datetime64', nulls: 0.0 },
    { col: 'FEDFUNDS', dtype: 'float64', nulls: 0.0 },
    { col: 'CPIAUCSL', dtype: 'float64', nulls: 1.2 },
    { col: 'UNRATE', dtype: 'float64', nulls: 0.4 },
  ],
};
const DEFAULT_SCHEMA = [
  { col: 'date', dtype: 'datetime64', nulls: 0.0 },
  { col: 'value', dtype: 'float64', nulls: 1.0 },
  { col: 'category', dtype: 'object', nulls: 0.0 },
];

const RAW_PREVIEW = {
  'market_commodities.parquet': {
    cols: ['date', 'copper', 'steel', 'aluminum'],
    rows: [
      ['2026-06-01', '8512.40', '648.20', '2204.1'],
      ['2026-05-01', '8488.10', '651.05', '2189.7'],
      ['2026-04-01', '8401.55', '643.80', '2176.3'],
      ['2026-03-01', '8377.20', '639.42', '2160.9'],
      ['2026-02-01', '8290.00', '634.10', '2148.5'],
    ],
  },
  'fred_macro.parquet': {
    cols: ['date', 'FEDFUNDS', 'CPIAUCSL', 'UNRATE'],
    rows: [
      ['2026-06-01', '5.375', '312.4', '3.7'],
      ['2026-05-01', '5.375', '311.8', '3.8'],
      ['2026-04-01', '5.375', '311.1', '3.7'],
      ['2026-03-01', '5.500', '310.5', '3.9'],
      ['2026-02-01', '5.500', '309.9', '3.8'],
    ],
  },
};
const DEFAULT_RAW = {
  cols: ['date', 'value', 'category'],
  rows: [
    ['2026-06-01', '128.4', 'A'],
    ['2026-05-01', '126.9', 'A'],
    ['2026-04-01', '125.1', 'B'],
    ['2026-03-01', '124.7', 'B'],
    ['2026-02-01', '123.2', 'A'],
  ],
};

export default function DataExplorer() {
  const { user } = useAuth();
  const canRaw = can(user, PERMISSIONS.VIEW_RAW_DATA);

  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState('all');

  const filtered = filter === 'all' ? DATASETS : DATASETS.filter(d => d.type === filter);
  const sel = DATASETS.find(d => d.name === selected);
  const schema = sel ? (SCHEMAS[sel.name] || DEFAULT_SCHEMA) : [];
  const raw = sel ? (RAW_PREVIEW[sel.name] || DEFAULT_RAW) : null;

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

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Data Explorer</h1>
          <p className="text-slate-400 text-sm mt-1">Dataset catalog · Schema · Data quality report · Source tracing</p>
        </div>
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${canRaw ? 'border-emerald-700 text-emerald-300 bg-emerald-900/20' : 'border-slate-600 text-slate-400 bg-slate-700/30'}`}>
          {canRaw ? 'Raw data access' : '🔒 Aggregated view (raw data is Admin-only)'}
        </span>
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

              {/* Data quality bar (aggregated — always visible) */}
              <div className="mt-4">
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Completeness</span>
                  <span className={sel.nullPct > 5 ? 'text-red-400' : sel.nullPct > 1 ? 'text-yellow-400' : 'text-green-400'}>
                    {(100 - sel.nullPct).toFixed(1)}%
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-slate-700">
                  <div className="h-1.5 rounded-full" style={{ width: `${100 - sel.nullPct}%`, backgroundColor: sel.nullPct > 5 ? '#ef4444' : sel.nullPct > 1 ? '#f59e0b' : '#22c55e' }} />
                </div>
              </div>

              {/* Schema (aggregated — always visible) */}
              <div className="mt-4">
                <p className="text-slate-400 text-xs mb-2">Schema</p>
                <div className="space-y-1">
                  {schema.map((s) => (
                    <div key={s.col} className="flex items-center justify-between text-xs">
                      <code className="text-slate-200">{s.col}</code>
                      <span className="flex items-center gap-2">
                        <span className="text-slate-500 font-mono">{s.dtype}</span>
                        <span className={s.nulls > 1 ? 'text-yellow-400' : 'text-green-400'}>{s.nulls}% null</span>
                      </span>
                    </div>
                  ))}
                  {schema.length < sel.cols && (
                    <p className="text-slate-600 text-[11px] mt-1">+ {sel.cols - schema.length} more column(s)</p>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-48 text-slate-500 text-sm text-center">
              <p className="text-2xl mb-3">🗄️</p>
              <p>Click a dataset row to see details, schema and quality</p>
            </div>
          )}
        </div>
      </div>

      {/* Raw data preview — gated behind view_raw_data (Admin) */}
      {sel && (
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Raw Data Preview — <span className="font-mono text-base">{sel.name}</span></h2>
              <p className="text-slate-500 text-xs mt-1">First rows of the underlying file</p>
            </div>
            <Badge label={canRaw ? 'Admin access' : 'Admin only'} color={canRaw ? 'green' : 'slate'} />
          </div>

          {canRaw ? (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-700">
                    {raw.cols.map((c) => <th key={c} className="text-left px-3 py-2 font-mono">{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {raw.rows.map((row, i) => (
                    <tr key={i} className="border-b border-slate-800 hover:bg-slate-800/40">
                      {row.map((cell, j) => <td key={j} className="px-3 py-1.5 font-mono text-slate-200">{cell}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-slate-600 text-[11px] mt-2">Showing {raw.rows.length} of {sel.rows.toLocaleString()} rows.</p>
            </div>
          ) : (
            <div className="rounded-lg p-8 border border-dashed border-slate-600 text-center" style={{ backgroundColor: '#0f172a' }}>
              <p className="text-3xl mb-2">🔒</p>
              <p className="text-slate-300 text-sm font-medium">Raw data is Admin-only</p>
              <p className="text-slate-500 text-xs mt-1 max-w-md mx-auto">
                Your role can view aggregated statistics, schema and data-quality metrics. Full row-level data
                requires the <code className="text-blue-400">view_raw_data</code> permission. Sign in as Administrator to preview raw rows.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Scripts Reference */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-3">Data Scripts</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {[
            { cmd: 'python scripts/fetch_data.py',           desc: 'Fetch real data from Yahoo Finance, FRED/ONS, LME/Fastmarkets' },
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
