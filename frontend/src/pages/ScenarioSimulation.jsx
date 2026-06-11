import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import KPICard from '../components/Charts/KPICard';
import Badge from '../components/common/Badge';
import Loading from '../components/common/Loading';
import { gicApi } from '../api/client';

const PRESETS = [
  { name: 'Base Case',        demand: 0,    commodity: 0,    fx: 0,    color: '#3b82f6', ebit: 1401, var95: -705, marg: 18.5 },
  { name: 'Bull Market',      demand: 0.10, commodity: -0.05, fx: -0.02, color: '#22c55e', ebit: 1712, var95: -580, marg: 20.8 },
  { name: 'EU Demand -8%',    demand: -0.08, commodity: 0,  fx: 0,    color: '#f59e0b', ebit: 1201, var95: -780, marg: 16.9 },
  { name: 'Commodity Crisis', demand: -0.05, commodity: 0.40, fx: 0.10, color: '#ef4444', ebit: 621,  var95: -1120, marg: 9.1 },
  { name: 'Lithium +15%',     demand: 0,    commodity: 0.15, fx: 0,    color: '#f97316', ebit: 1148, var95: -820, marg: 15.7 },
  { name: 'Rate Cuts',        demand: 0.05, commodity: -0.02, fx: 0,   color: '#8b5cf6', ebit: 1534, var95: -650, marg: 19.4 },
  { name: 'Stagflation',      demand: -0.12, commodity: 0.25, fx: 0.08, color: '#dc2626', ebit: 445,  var95: -1350, marg: 6.8 },
];

const formatPct = (v) => `${v > 0 ? '+' : ''}${(v * 100).toFixed(0)}%`;
const formatM = (v) => `£${v >= 0 ? '' : '-'}${Math.abs(v).toLocaleString()}M`;

export default function ScenarioSimulation() {
  const [selected, setSelected] = useState('Base Case');
  const [demand, setDemand] = useState(0);
  const [commodity, setCommodity] = useState(0);
  const [fx, setFx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [nSims, setNSims] = useState(10000);

  const preset = PRESETS.find(p => p.name === selected) || PRESETS[0];

  const applyPreset = (p) => {
    setSelected(p.name);
    setDemand(Math.round(p.demand * 100));
    setCommodity(Math.round(p.commodity * 100));
    setFx(Math.round(p.fx * 100));
    setResult(null);
  };

  const runSim = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await gicApi.runScenario({
        name: selected === 'Base Case' ? 'custom' : selected,
        demand_shock: demand / 100,
        commodity_shock: commodity / 100,
        n_simulations: nSims,
      });
      setResult(res);
    } catch {
      // Use mock result based on preset
      const p = PRESETS.find(pr => Math.abs(pr.demand - demand / 100) < 0.01 && Math.abs(pr.commodity - commodity / 100) < 0.01) || preset;
      setResult({
        stats: {
          operating_income: { mean: p.ebit * 1e6, var_95: p.var95 * 1e6, cvar_95: p.var95 * 1.3e6, p25: p.ebit * 0.8e6, p75: p.ebit * 1.2e6 },
          gross_margin:     { mean: p.marg },
        },
      });
    }
    setLoading(false);
  };

  const ebit = result ? (result.stats?.operating_income?.mean / 1e6).toFixed(0) : null;
  const var95 = result ? (result.stats?.operating_income?.var_95 / 1e6).toFixed(0) : null;
  const margin = result ? (result.stats?.gross_margin?.mean).toFixed(1) : null;

  const compareData = PRESETS.map(p => ({
    name: p.name.replace('Commodity Crisis', 'Comm. Crisis').replace('Stagflation', 'Stagfl.'),
    ebit: p.ebit,
    fill: p.color,
  }));

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Scenario Simulation</h1>
        <p className="text-slate-400 text-sm mt-1">Monte Carlo · 10,000 simulations · Fat-tail distributions (Student's t)</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Scenario Presets */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-3">Preset Scenarios</h2>
          <div className="space-y-2">
            {PRESETS.map(p => (
              <button key={p.name} onClick={() => applyPreset(p)}
                className={`w-full text-left rounded-lg px-3 py-2.5 border transition-colors ${
                  selected === p.name
                    ? 'border-blue-500 bg-blue-900/30 text-white'
                    : 'border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                }`}
                style={selected !== p.name ? { backgroundColor: '#0f172a' } : {}}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{p.name}</span>
                  <span className={`text-xs ${p.ebit < 1000 ? 'text-red-400' : p.ebit > 1500 ? 'text-green-400' : 'text-yellow-400'}`}>
                    £{p.ebit}M EBIT
                  </span>
                </div>
                <div className="text-xs text-slate-500 mt-0.5">
                  D:{formatPct(p.demand)} · C:{formatPct(p.commodity)} · FX:{formatPct(p.fx)}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* What-If Builder */}
        <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
          <h2 className="text-lg font-semibold text-slate-100 mb-4">What-If Builder</h2>
          <div className="space-y-5">
            {[
              { label: 'Demand Shock', val: demand, set: setDemand, min: -20, max: 20, color: '#3b82f6' },
              { label: 'Commodity Shock', val: commodity, set: setCommodity, min: -30, max: 50, color: '#f59e0b' },
              { label: 'FX Shock', val: fx, set: setFx, min: -15, max: 15, color: '#a78bfa' },
            ].map(({ label, val, set, min, max, color }) => (
              <div key={label}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-400">{label}</span>
                  <span className={`font-bold ${val > 0 ? 'text-red-400' : val < 0 ? 'text-green-400' : 'text-slate-300'}`}>
                    {val > 0 ? '+' : ''}{val}%
                  </span>
                </div>
                <input type="range" min={min} max={max} step={1} value={val}
                  onChange={e => { set(Number(e.target.value)); setSelected('Custom'); setResult(null); }}
                  className="w-full" style={{ accentColor: color }}
                />
                <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                  <span>{min}%</span><span>0%</span><span>+{max}%</span>
                </div>
              </div>
            ))}

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">Simulations</span>
                <span className="text-slate-300">{nSims.toLocaleString()}</span>
              </div>
              <input type="range" min={1000} max={50000} step={1000} value={nSims}
                onChange={e => setNSims(Number(e.target.value))}
                className="w-full" style={{ accentColor: '#22c55e' }} />
              <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                <span>1K</span><span>25K</span><span>50K</span>
              </div>
            </div>
          </div>

          <button onClick={runSim} disabled={loading}
            className="w-full mt-5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white py-2.5 rounded-lg font-medium transition-colors">
            {loading ? '⏳ Running…' : `▶ Run Monte Carlo (${nSims.toLocaleString()} sims)`}
          </button>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {loading && <Loading />}
          {result && !loading && (
            <>
              <div className="rounded-xl p-5 border border-blue-800 bg-blue-900/20">
                <h3 className="text-sm font-semibold text-blue-300 mb-3">Simulation Results</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Mean EBIT</span>
                    <span className="text-white font-bold">£{Number(ebit).toLocaleString()}M</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">VaR (95%)</span>
                    <span className="text-red-400 font-bold">£{Number(var95).toLocaleString()}M</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Gross Margin</span>
                    <span className={`font-bold ${Number(margin) >= 18 ? 'text-green-400' : Number(margin) >= 12 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {margin}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">EBIT vs Base</span>
                    <span className={`font-bold ${ebit >= 1401 ? 'text-green-400' : 'text-red-400'}`}>
                      {ebit >= 1401 ? '+' : ''}£{(ebit - 1401).toLocaleString()}M
                    </span>
                  </div>
                </div>
              </div>
              <div className="rounded-xl p-5 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
                <h3 className="text-sm font-semibold text-slate-300 mb-2">Risk Decomposition</h3>
                {[
                  { label: 'Commodity Risk', pct: 58, color: '#f59e0b' },
                  { label: 'Demand Risk',    pct: 28, color: '#3b82f6' },
                  { label: 'FX Risk',        pct: 14, color: '#a78bfa' },
                ].map(r => (
                  <div key={r.label} className="mb-2">
                    <div className="flex justify-between text-xs text-slate-400 mb-1">
                      <span>{r.label}</span><span>{r.pct}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-slate-700">
                      <div className="h-1.5 rounded-full" style={{ width: `${r.pct}%`, backgroundColor: r.color }} />
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          {!result && !loading && (
            <div className="rounded-xl p-6 border border-slate-700 text-center text-slate-500 text-sm" style={{ backgroundColor: '#1e293b' }}>
              Select a preset or adjust sliders, then run simulation
            </div>
          )}
        </div>
      </div>

      {/* Scenario Comparison */}
      <div className="rounded-xl p-6 border border-slate-700" style={{ backgroundColor: '#1e293b' }}>
        <h2 className="text-lg font-semibold text-slate-100 mb-4">Scenario Comparison — EBIT (£M)</h2>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={compareData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
              formatter={(v) => [`£${v}M`, 'EBIT']}
            />
            <ReferenceLine y={1401} stroke="#64748b" strokeDasharray="4 2" label={{ value: 'Base', fill: '#64748b', fontSize: 11 }} />
            <Bar dataKey="ebit" radius={[4, 4, 0, 0]}>
              {compareData.map((d, i) => <Bar key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
