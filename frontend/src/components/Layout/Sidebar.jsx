import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/executive', label: 'Executive Summary', emoji: '📊' },
  { path: '/commodity', label: 'Commodity Intelligence', emoji: '📈' },
  { path: '/pnl', label: 'Financial P&L', emoji: '💰' },
  { path: '/simulation', label: 'Scenario Simulation', emoji: '🎲' },
  { path: '/market', label: 'Market Monitor', emoji: '🌐' },
  { path: '/governance', label: 'Governance & LLM', emoji: '🛡️' },
  { path: '/data', label: 'Data Explorer', emoji: '🗄️' },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div
      className="flex flex-col h-full transition-all duration-300 border-r border-slate-700"
      style={{
        width: collapsed ? '64px' : '256px',
        backgroundColor: '#1e293b',
        minWidth: collapsed ? '64px' : '256px',
      }}
    >
      {/* Logo / Branding */}
      <div className="flex items-center justify-between px-4 py-5 border-b border-slate-700">
        {!collapsed && (
          <div>
            <div className="text-lg font-bold text-white tracking-tight">GIC Platform</div>
            <div className="text-xs text-slate-400 mt-0.5">Financial Intelligence</div>
          </div>
        )}
        {collapsed && (
          <div className="w-full flex justify-center">
            <span className="text-2xl">📡</span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-slate-400 hover:text-white transition-colors ml-2"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            {collapsed ? (
              <path d="M6 12l4-4-4-4v8z" />
            ) : (
              <path d="M10 12l-4-4 4-4v8z" />
            )}
          </svg>
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto scrollbar-thin">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 group ${
                isActive
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700'
              }`
            }
            title={collapsed ? item.label : ''}
          >
            <span className="text-base flex-shrink-0">{item.emoji}</span>
            {!collapsed && (
              <span className="truncate">{item.label}</span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-slate-700">
        {!collapsed ? (
          <div>
            <div className="text-xs text-slate-500 font-medium">Version</div>
            <div className="text-xs text-slate-400 mt-0.5">v1.0.0 · Plan-to-Perform</div>
            <div className="mt-3 flex items-center gap-2">
              <span
                className="inline-block w-2 h-2 rounded-full pulse-dot"
                style={{ backgroundColor: '#22c55e' }}
              />
              <span className="text-xs text-slate-400">Engine Active</span>
            </div>
          </div>
        ) : (
          <div className="flex justify-center">
            <span
              className="inline-block w-2 h-2 rounded-full pulse-dot"
              style={{ backgroundColor: '#22c55e' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
