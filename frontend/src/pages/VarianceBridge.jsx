import React from 'react';

export default function VarianceBridge() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-4 rounded-xl border border-blue-800/50 bg-blue-900/20 px-4 py-3 flex items-center gap-3">
        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
        <span className="text-sm text-blue-200">
          Connecting to backend variance engine (
          <code className="text-blue-300">/insights/variance-bridge</code>)…
        </span>
      </div>

      <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-10 text-center">
        <div className="text-5xl mb-4">🌉</div>
        <h2 className="text-2xl font-bold text-white mb-2">Plan-to-Perform Variance Bridge</h2>
        <p className="text-slate-400 max-w-md mx-auto leading-relaxed">
          Coming together… A driver-based bridge decomposing plan-vs-actual into volume, price, mix
          and FX, so the story of every gap is unambiguous.
        </p>
      </div>
    </div>
  );
}
