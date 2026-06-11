import React from 'react';

export default function InsightsCenter() {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-4 rounded-xl border border-blue-800/50 bg-blue-900/20 px-4 py-3 flex items-center gap-3">
        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
        <span className="text-sm text-blue-200">
          Connecting to backend insights feed (<code className="text-blue-300">/insights/feed</code>,{' '}
          <code className="text-blue-300">/insights/early-warning</code>)…
        </span>
      </div>

      <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-10 text-center">
        <div className="text-5xl mb-4">💡</div>
        <h2 className="text-2xl font-bold text-white mb-2">Insights Center</h2>
        <p className="text-slate-400 max-w-md mx-auto leading-relaxed">
          Coming together… Prescriptive insights, ranked recommendations and early-warning signals
          will surface here, tied to quantified financial impact.
        </p>
      </div>
    </div>
  );
}
