import React, { useEffect, useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { gicApi } from '../api/client';

const FALLBACK_PROFILES = [
  {
    label: 'Administrator',
    username: 'admin',
    role: 'ADMIN',
    description: 'Full access — simulation, thresholds, retraining, governance and user control.',
    demo_password: 'admin123',
  },
  {
    label: 'Analyst (Viewer)',
    username: 'user',
    role: 'USER',
    description: 'Explore dashboards, forecasts and insights; run sandbox simulations.',
    demo_password: 'user123',
  },
];

const TRUST_BULLETS = [
  'Calibrated, probabilistic forecasting',
  '10,000-path Monte-Carlo risk engine',
  'Explainable AI with full audit trail',
  'Enterprise role-based access control',
];

function pickProfile(profiles, role) {
  return (
    profiles.find((p) => (p.role || '').toUpperCase() === role) ||
    FALLBACK_PROFILES.find((p) => p.role === role)
  );
}

export default function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, loginWithDemo } = useAuth();

  const redirectTo = location.state?.from?.pathname || '/app/executive';

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [demoLoading, setDemoLoading] = useState('');
  const [profiles, setProfiles] = useState(FALLBACK_PROFILES);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await gicApi.demoProfiles();
        if (!cancelled && Array.isArray(data) && data.length) setProfiles(data);
      } catch {
        // keep fallback profiles
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      await login(username.trim(), password);
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(err?.message || 'Invalid username or password.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleDemo = async (role) => {
    setError('');
    setDemoLoading(role);
    try {
      const profile = pickProfile(profiles, role);
      if (!profile) throw new Error('Demo profile unavailable.');
      // Ensure a password is present (demo_password may be absent outside demo mode).
      const withPwd = profile.demo_password
        ? profile
        : { ...profile, demo_password: pickProfile(FALLBACK_PROFILES, role)?.demo_password };
      await loginWithDemo(withPwd);
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(err?.message || 'Demo sign-in failed.');
    } finally {
      setDemoLoading('');
    }
  };

  const adminProfile = pickProfile(profiles, 'ADMIN');
  const userProfile = pickProfile(profiles, 'USER');

  return (
    <div className="min-h-screen flex" style={{ backgroundColor: '#0f172a' }}>
      <style>{`
        .gic-grid-bg{background-image:radial-gradient(circle at 1px 1px, rgba(255,255,255,.06) 1px, transparent 0);background-size:34px 34px;}
      `}</style>

      {/* Left brand panel */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden bg-gradient-to-br from-blue-900 via-slate-900 to-slate-950">
        <div className="absolute inset-0 gic-grid-bg opacity-50" />
        <div
          className="absolute -top-32 -left-20 w-96 h-96 rounded-full blur-3xl"
          style={{ background: 'radial-gradient(circle, rgba(59,130,246,.35), transparent 70%)' }}
        />
        <div
          className="absolute -bottom-32 -right-10 w-96 h-96 rounded-full blur-3xl"
          style={{ background: 'radial-gradient(circle, rgba(16,185,129,.22), transparent 70%)' }}
        />
        <div className="relative z-10 flex flex-col justify-between p-12 w-full">
          <Link to="/" className="flex items-center gap-3 w-fit">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-xl shadow-lg shadow-blue-900/40">
              📡
            </div>
            <div>
              <div className="text-lg font-bold text-white leading-none">GIC Intelligence</div>
              <div className="text-[10px] text-blue-200/70 tracking-widest uppercase">Plan to Perform</div>
            </div>
          </Link>

          <div>
            <h2 className="text-4xl font-extrabold text-white leading-tight">
              From plan to performance,
              <br />
              <span className="bg-gradient-to-r from-blue-300 to-cyan-200 bg-clip-text text-transparent">
                intelligently.
              </span>
            </h2>
            <p className="mt-5 text-blue-100/70 max-w-md leading-relaxed">
              Sign in to a unified intelligence layer for calibrated forecasts, quantified risk and
              prescriptive action.
            </p>
            <ul className="mt-8 space-y-3">
              {TRUST_BULLETS.map((b) => (
                <li key={b} className="flex items-center gap-3 text-blue-50/80 text-sm">
                  <span className="w-5 h-5 rounded-full bg-emerald-500/20 border border-emerald-400/40 flex items-center justify-center text-emerald-300 text-xs">
                    ✓
                  </span>
                  {b}
                </li>
              ))}
            </ul>
          </div>

          <div className="text-xs text-blue-200/40">
            © {new Date().getFullYear()} GIC Financial Intelligence
          </div>
        </div>
      </div>

      {/* Right login card */}
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          <div className="lg:hidden mb-8 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-lg">
              📡
            </div>
            <div className="text-lg font-bold text-white">GIC Intelligence</div>
          </div>

          <div className="rounded-2xl border border-slate-700 bg-slate-800/60 backdrop-blur p-8 shadow-2xl">
            <h1 className="text-2xl font-bold text-white">Welcome back</h1>
            <p className="text-sm text-slate-400 mt-1">Sign in to access the platform.</p>

            {error && (
              <div className="mt-5 rounded-lg border border-red-800 bg-red-900/30 px-4 py-3 text-sm text-red-300">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="mt-6 space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Username</label>
                <input
                  type="text"
                  autoComplete="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full rounded-lg bg-slate-900 border border-slate-700 px-3.5 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                  placeholder="Enter your username"
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Password</label>
                <input
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-lg bg-slate-900 border border-slate-700 px-3.5 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                  placeholder="Enter your password"
                  required
                />
              </div>
              <button
                type="submit"
                disabled={submitting}
                className="w-full rounded-lg bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold py-2.5 text-sm shadow-lg shadow-blue-900/40 transition-all"
              >
                {submitting ? 'Signing in…' : 'Sign In'}
              </button>
            </form>

            {/* Quick demo access */}
            <div className="mt-7">
              <div className="flex items-center gap-3 mb-4">
                <div className="flex-1 h-px bg-slate-700" />
                <span className="text-[11px] uppercase tracking-widest text-slate-500 font-medium">
                  Quick Demo Access
                </span>
                <div className="flex-1 h-px bg-slate-700" />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <button
                  onClick={() => handleDemo('ADMIN')}
                  disabled={!!demoLoading}
                  className="text-left rounded-xl border border-blue-800/60 bg-blue-900/20 hover:bg-blue-900/40 p-4 transition disabled:opacity-60"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span>👑</span>
                    <span className="font-semibold text-white text-sm">Administrator</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-snug">
                    {adminProfile?.description || 'Full platform access and controls.'}
                  </p>
                  <span className="mt-2 inline-block text-[11px] font-medium text-blue-300">
                    {demoLoading === 'ADMIN' ? 'Signing in…' : 'Enter as Admin →'}
                  </span>
                </button>

                <button
                  onClick={() => handleDemo('USER')}
                  disabled={!!demoLoading}
                  className="text-left rounded-xl border border-slate-700 bg-slate-900/40 hover:bg-slate-700/40 p-4 transition disabled:opacity-60"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span>🔍</span>
                    <span className="font-semibold text-white text-sm">Analyst (Viewer)</span>
                  </div>
                  <p className="text-[11px] text-slate-400 leading-snug">
                    {userProfile?.description || 'Read-only exploration and sandbox simulation.'}
                  </p>
                  <span className="mt-2 inline-block text-[11px] font-medium text-slate-300">
                    {demoLoading === 'USER' ? 'Signing in…' : 'Enter as Analyst →'}
                  </span>
                </button>
              </div>
            </div>
          </div>

          <div className="mt-6 text-center">
            <Link to="/" className="text-sm text-slate-400 hover:text-white transition-colors">
              ← Back to home
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
