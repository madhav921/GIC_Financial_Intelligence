import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const STATS = [
  { label: 'Revenue Modeled', target: 19.8, suffix: 'B', prefix: '£', decimals: 1 },
  { label: 'Monte-Carlo Sims', target: 10000, suffix: '', prefix: '', decimals: 0 },
  { label: 'CI Calibration', target: 79, suffix: '%', prefix: '', decimals: 0 },
  { label: 'Scenario Recalc', target: 1, suffix: 's', prefix: '<', decimals: 0 },
];

const FEATURES = [
  {
    icon: '🔮',
    title: 'Predictive Forecasting',
    desc: 'Calibrated, probabilistic forecasts across drivers and horizons — not single-point guesses, but ranges with quantified confidence.',
  },
  {
    icon: '💰',
    title: 'Driver-Based P&L',
    desc: 'A live, driver-linked P&L that traces every line back to the operational and market signals that move it.',
  },
  {
    icon: '🎲',
    title: 'Monte-Carlo Risk',
    desc: '10,000-path simulations translate uncertainty into distributions, VaR, and downside-aware planning.',
  },
  {
    icon: '🧠',
    title: 'Prescriptive Insights',
    desc: 'From "what happened" to "what to do" — ranked, explainable actions tied to financial impact.',
  },
  {
    icon: '🌉',
    title: 'Plan-to-Perform Variance Bridge',
    desc: 'Decompose plan-vs-actual into volume, price, mix, and FX so the story of the gap is unambiguous.',
  },
  {
    icon: '🛡️',
    title: 'Governance & Explainable AI',
    desc: 'Full audit trails, model lineage, and human-readable narratives — intelligence you can defend.',
  },
];

const STEPS = [
  { n: '01', icon: '🗄️', title: 'Data', desc: 'Ingest market, operational and financial signals.' },
  { n: '02', icon: '🔮', title: 'Intelligence', desc: 'Forecast drivers with calibrated confidence.' },
  { n: '03', icon: '💰', title: 'Financial', desc: 'Roll drivers into a live driver-based P&L.' },
  { n: '04', icon: '🎲', title: 'Simulation', desc: 'Stress-test plans across thousands of paths.' },
  { n: '05', icon: '🛡️', title: 'Governance', desc: 'Explain, audit and sign off with confidence.' },
];

function useCountUp(target, decimals, run) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!run) return undefined;
    let raf;
    const start = performance.now();
    const dur = 1400;
    const step = (t) => {
      const p = Math.min((t - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(target * eased);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, run]);
  return decimals === 0 ? Math.round(val).toLocaleString() : val.toFixed(decimals);
}

function StatItem({ stat, run }) {
  const value = useCountUp(stat.target, stat.decimals, run);
  return (
    <div className="text-center px-6 py-4">
      <div className="text-3xl md:text-4xl font-extrabold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent tabular-nums">
        {stat.prefix}
        {value}
        {stat.suffix}
      </div>
      <div className="mt-1 text-xs md:text-sm text-slate-400 font-medium uppercase tracking-wider">
        {stat.label}
      </div>
    </div>
  );
}

export default function Landing() {
  const navigate = useNavigate();
  const [statsVisible, setStatsVisible] = useState(false);
  const statsRef = useRef(null);

  useEffect(() => {
    const el = statsRef.current;
    if (!el) return undefined;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setStatsVisible(true);
      },
      { threshold: 0.3 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div className="min-h-screen text-slate-200" style={{ backgroundColor: '#0f172a' }}>
      <style>{`
        @keyframes gicFloat {0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
        @keyframes gicGlow {0%,100%{opacity:.35}50%{opacity:.65}}
        .gic-grid-bg{background-image:radial-gradient(circle at 1px 1px, rgba(59,130,246,.18) 1px, transparent 0);background-size:38px 38px;}
        .gic-card{transition:transform .25s ease, box-shadow .25s ease, border-color .25s ease;}
        .gic-card:hover{transform:translateY(-6px);border-color:rgba(59,130,246,.5);box-shadow:0 18px 40px -12px rgba(59,130,246,.35);}
      `}</style>

      {/* Top nav */}
      <header className="sticky top-0 z-40 border-b border-slate-800/80 backdrop-blur-md bg-slate-900/70">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-lg shadow-lg shadow-blue-900/40">
              📡
            </div>
            <div>
              <div className="font-bold text-white tracking-tight leading-none">GIC Intelligence</div>
              <div className="text-[10px] text-slate-400 tracking-widest uppercase">Plan to Perform</div>
            </div>
          </div>
          <nav className="hidden md:flex items-center gap-7 text-sm text-slate-300">
            <a href="#capabilities" className="hover:text-white transition-colors">Capabilities</a>
            <a href="#how" className="hover:text-white transition-colors">How it works</a>
            <a href="#security" className="hover:text-white transition-colors">Security</a>
          </nav>
          <button
            onClick={() => navigate('/login')}
            className="px-4 py-2 rounded-lg text-sm font-semibold bg-blue-600 hover:bg-blue-500 text-white transition-colors shadow-lg shadow-blue-900/40"
          >
            Sign In
          </button>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 gic-grid-bg opacity-60" />
        <div
          className="absolute -top-40 -right-32 w-[36rem] h-[36rem] rounded-full blur-3xl"
          style={{ background: 'radial-gradient(circle, rgba(59,130,246,.25), transparent 70%)', animation: 'gicGlow 6s ease-in-out infinite' }}
        />
        <div
          className="absolute -bottom-40 -left-32 w-[32rem] h-[32rem] rounded-full blur-3xl"
          style={{ background: 'radial-gradient(circle, rgba(16,185,129,.18), transparent 70%)', animation: 'gicGlow 7s ease-in-out infinite' }}
        />
        <div className="relative max-w-7xl mx-auto px-6 pt-24 pb-20 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-slate-800/80 border border-slate-700 text-slate-300 mb-7">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Financial Plan-to-Perform Intelligence
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-white leading-[1.05]">
            From Plan to Performance,
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-cyan-300 to-emerald-300 bg-clip-text text-transparent">
              Intelligently.
            </span>
          </h1>
          <p className="mt-7 max-w-2xl mx-auto text-lg md:text-xl text-slate-400 leading-relaxed">
            Turn raw financial data into calibrated forecasts, quantified risk, and prescriptive
            action — a unified intelligence layer that connects every plan to its outcome.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate('/login')}
              className="px-7 py-3.5 rounded-xl text-base font-semibold bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white shadow-xl shadow-blue-900/40 transition-all"
            >
              Launch Platform →
            </button>
            <a
              href="#capabilities"
              className="px-7 py-3.5 rounded-xl text-base font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
            >
              View Capabilities
            </a>
          </div>
        </div>
      </section>

      {/* Stat band */}
      <section ref={statsRef} className="border-y border-slate-800 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-6 py-6 grid grid-cols-2 md:grid-cols-4 divide-x divide-slate-800">
          {STATS.map((s) => (
            <StatItem key={s.label} stat={s} run={statsVisible} />
          ))}
        </div>
      </section>

      {/* Features */}
      <section id="capabilities" className="max-w-7xl mx-auto px-6 py-24">
        <div className="text-center max-w-2xl mx-auto mb-14">
          <div className="text-sm font-semibold text-blue-400 uppercase tracking-widest mb-3">Capabilities</div>
          <h2 className="text-3xl md:text-4xl font-bold text-white">One platform, the full decision loop</h2>
          <p className="mt-4 text-slate-400">
            Forecast, plan, simulate and govern — every capability sharing one driver model and one
            source of truth.
          </p>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="gic-card rounded-xl border border-slate-700 bg-slate-800/60 backdrop-blur p-6"
            >
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500/20 to-cyan-400/10 border border-slate-700 flex items-center justify-center text-2xl mb-4">
                {f.icon}
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="border-y border-slate-800 bg-slate-900/40">
        <div className="max-w-7xl mx-auto px-6 py-24">
          <div className="text-center max-w-2xl mx-auto mb-14">
            <div className="text-sm font-semibold text-blue-400 uppercase tracking-widest mb-3">How it works</div>
            <h2 className="text-3xl md:text-4xl font-bold text-white">A pipeline from signal to sign-off</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {STEPS.map((s, i) => (
              <div key={s.n} className="relative">
                <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-5 h-full">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-2xl">{s.icon}</span>
                    <span className="text-xs font-bold text-slate-600">{s.n}</span>
                  </div>
                  <div className="font-semibold text-white">{s.title}</div>
                  <div className="text-xs text-slate-400 mt-1 leading-relaxed">{s.desc}</div>
                </div>
                {i < STEPS.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 -right-2 text-slate-600 z-10">→</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Security / roles */}
      <section id="security" className="max-w-7xl mx-auto px-6 py-24">
        <div className="rounded-2xl border border-slate-700 bg-gradient-to-br from-slate-800/80 to-slate-900 p-10 md:p-14 relative overflow-hidden">
          <div
            className="absolute -top-24 -right-24 w-80 h-80 rounded-full blur-3xl"
            style={{ background: 'radial-gradient(circle, rgba(59,130,246,.2), transparent 70%)' }}
          />
          <div className="relative grid md:grid-cols-2 gap-10 items-center">
            <div>
              <div className="text-sm font-semibold text-emerald-400 uppercase tracking-widest mb-3">
                Enterprise-grade
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Role-based access, built in
              </h2>
              <p className="text-slate-400 leading-relaxed mb-6">
                Every capability is gated by granular permissions. Analysts explore and model;
                administrators control thresholds, retraining and governance — with a complete audit
                trail behind every action.
              </p>
              <div className="flex flex-wrap gap-2">
                {['RBAC', 'Audit Trail', 'Explainable AI', 'Token Auth', 'Least Privilege'].map((t) => (
                  <span
                    key={t}
                    className="px-3 py-1 rounded-full text-xs font-medium bg-slate-800 border border-slate-700 text-slate-300"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4">
              <div className="rounded-xl border border-blue-800/60 bg-blue-900/20 p-5">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">👑</span>
                  <span className="font-semibold text-white">Administrator</span>
                </div>
                <p className="text-xs text-slate-400">
                  Full access: simulation, threshold management, retraining, governance and user
                  control.
                </p>
              </div>
              <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-5">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🔍</span>
                  <span className="font-semibold text-white">Analyst (Viewer)</span>
                </div>
                <p className="text-xs text-slate-400">
                  Explore dashboards, forecasts and insights; run sandbox simulations within guardrails.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-7xl mx-auto px-6 pb-24">
        <div className="text-center rounded-2xl border border-slate-700 bg-slate-800/40 p-12">
          <h2 className="text-3xl md:text-4xl font-bold text-white">Ready to see your plan perform?</h2>
          <p className="mt-4 text-slate-400 max-w-xl mx-auto">
            Launch the platform and explore live forecasts, simulations and prescriptive insights.
          </p>
          <button
            onClick={() => navigate('/login')}
            className="mt-8 px-8 py-3.5 rounded-xl text-base font-semibold bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white shadow-xl shadow-blue-900/40 transition-all"
          >
            Launch Platform →
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/60">
        <div className="max-w-7xl mx-auto px-6 py-10 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center">
              📡
            </div>
            <div className="text-sm text-slate-400">
              <span className="text-white font-semibold">GIC Intelligence</span> · Plan-to-Perform Engine
            </div>
          </div>
          <div className="text-xs text-slate-500">
            © {new Date().getFullYear()} GIC Financial Intelligence. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
