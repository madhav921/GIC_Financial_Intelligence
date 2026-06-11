import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { gicApi } from '../../api/client';

const ROUTE_TITLES = {
  '/executive': 'Executive Summary',
  '/commodity': 'Commodity Intelligence',
  '/pnl': 'Financial P&L',
  '/simulation': 'Scenario Simulation',
  '/market': 'Market Monitor',
  '/governance': 'Governance & LLM',
  '/data': 'Data Explorer',
};

export default function Header() {
  const location = useLocation();
  const [now, setNow] = useState(new Date());
  const [apiStatus, setApiStatus] = useState('checking'); // 'healthy' | 'error' | 'checking'

  // Update clock every second
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Check API health on mount and every 30 seconds
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await gicApi.health();
        setApiStatus('healthy');
      } catch {
        setApiStatus('error');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const pageTitle = ROUTE_TITLES[location.pathname] || 'GIC Platform';

  const formatDate = (date) => {
    return date.toLocaleString('en-GB', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  };

  return (
    <header
      className="flex items-center justify-between px-6 border-b border-slate-700 flex-shrink-0"
      style={{ height: '56px', backgroundColor: '#1e293b' }}
    >
      {/* Left: page title */}
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold text-white">{pageTitle}</h1>
        {/* Live badge */}
        <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-green-900/40 text-green-400 border border-green-800">
          <span
            className="w-1.5 h-1.5 rounded-full pulse-dot"
            style={{ backgroundColor: '#22c55e' }}
          />
          Live
        </span>
      </div>

      {/* Right: clock + API status */}
      <div className="flex items-center gap-4">
        {/* API health indicator */}
        <div className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full"
            style={{
              backgroundColor:
                apiStatus === 'healthy'
                  ? '#22c55e'
                  : apiStatus === 'error'
                  ? '#ef4444'
                  : '#f59e0b',
            }}
          />
          <span className="text-xs text-slate-400">
            API{' '}
            {apiStatus === 'healthy'
              ? 'Connected'
              : apiStatus === 'error'
              ? 'Offline'
              : 'Checking'}
          </span>
        </div>

        {/* Separator */}
        <div className="w-px h-4 bg-slate-600" />

        {/* Clock */}
        <span className="text-xs text-slate-400 font-mono tabular-nums">
          {formatDate(now)}
        </span>

        {/* User icon placeholder */}
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white"
          style={{ backgroundColor: '#3b82f6' }}
          title="GIC Analyst"
        >
          G
        </div>
      </div>
    </header>
  );
}
