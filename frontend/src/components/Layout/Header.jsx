import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../auth/AuthContext';
import { isAdmin } from '../../auth/permissions';
import useRealtime from '../../hooks/useRealtime';

const ROUTE_TITLES = {
  '/app/executive': 'Executive Summary',
  '/app/commodity': 'Commodity Intelligence',
  '/app/pnl': 'Financial P&L',
  '/app/simulation': 'Scenario Simulation',
  '/app/market': 'Market Monitor',
  '/app/insights': 'Insights Center',
  '/app/variance': 'Plan-to-Perform Variance Bridge',
  '/app/warranty': 'Warranty Analytics',
  '/app/governance': 'Governance & LLM',
  '/app/data': 'Data Explorer',
};

function initialsOf(user) {
  const name = user?.full_name || user?.username || 'GIC';
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const { connected, source } = useRealtime();
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const pageTitle = ROUTE_TITLES[location.pathname] || 'GIC Intelligence';
  const admin = isAdmin(user);

  const handleLogout = () => {
    logout();
    navigate('/login', { replace: true });
  };

  const formatDate = (date) =>
    date.toLocaleString('en-GB', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });

  const liveColor = connected ? '#22c55e' : '#f59e0b';

  return (
    <header
      className="flex items-center justify-between px-6 border-b border-slate-700 flex-shrink-0"
      style={{ height: '56px', backgroundColor: '#1e293b' }}
    >
      {/* Left: page title + live */}
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold text-white">{pageTitle}</h1>
        <span
          className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border"
          style={{
            backgroundColor: connected ? 'rgba(34,197,94,.12)' : 'rgba(245,158,11,.12)',
            color: liveColor,
            borderColor: connected ? 'rgba(34,197,94,.4)' : 'rgba(245,158,11,.4)',
          }}
          title={source === 'live' ? 'Live WebSocket feed' : 'Simulated realtime feed'}
        >
          <span className="w-1.5 h-1.5 rounded-full pulse-dot" style={{ backgroundColor: liveColor }} />
          {connected ? 'Live' : 'Simulated'}
        </span>
      </div>

      {/* Right: clock + user + logout */}
      <div className="flex items-center gap-4">
        <span className="hidden md:inline text-xs text-slate-400 font-mono tabular-nums">
          {formatDate(now)}
        </span>

        <div className="w-px h-4 bg-slate-600" />

        {/* User */}
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white"
            style={{ backgroundColor: admin ? '#2563eb' : '#475569' }}
            title={user?.full_name || user?.username}
          >
            {initialsOf(user)}
          </div>
          <div className="hidden sm:block leading-tight">
            <div className="text-sm font-medium text-white">
              {user?.full_name || user?.username || 'Guest'}
            </div>
            <span
              className={`inline-block text-[10px] font-bold uppercase tracking-wider ${
                admin ? 'text-amber-400' : 'text-slate-400'
              }`}
            >
              {admin ? 'Administrator' : user?.role || 'Viewer'}
            </span>
          </div>
        </div>

        <button
          onClick={handleLogout}
          className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-slate-700/60 hover:bg-slate-600 hover:text-white border border-slate-600 transition-colors"
          title="Sign out"
        >
          Logout
        </button>
      </div>
    </header>
  );
}
