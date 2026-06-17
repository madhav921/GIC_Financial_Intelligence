import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../../auth/AuthContext';
import { can, isAdmin, PERMISSIONS } from '../../auth/permissions';

// Each item may declare a `permission`. If the user lacks it, the item is shown
// locked (and non-navigable) rather than hidden, so the RBAC story is visible.
const NAV_SECTIONS = [
  {
    title: 'Overview',
    items: [
      { path: '/app/executive', label: 'Executive Summary', emoji: '📊' },
      { path: '/app/market', label: 'Market Monitor', emoji: '🌐' },
    ],
  },
  {
    title: 'Intelligence',
    items: [
      { path: '/app/commodity', label: 'Commodity Intelligence', emoji: '📈' },
      { path: '/app/insights', label: 'Insights Center', emoji: '💡' },
      { path: '/app/warranty', label: 'Warranty Analytics', emoji: '🔧' },
    ],
  },
  {
    title: 'Planning',
    items: [
      { path: '/app/pnl', label: 'Financial P&L', emoji: '💰' },
      { path: '/app/variance', label: 'Variance Bridge', emoji: '🌉' },
      { path: '/app/simulation', label: 'Scenario Simulation', emoji: '🎲' },
    ],
  },
  {
    title: 'Governance',
    items: [
      { path: '/app/governance', label: 'Governance & LLM', emoji: '🛡️' },
      {
        path: '/app/data',
        label: 'Data Explorer',
        emoji: '🗄️',
        permission: PERMISSIONS.VIEW_AGGREGATED_DATA,
      },
    ],
  },
];

function RoleBadge({ user, collapsed }) {
  if (!user) return null;
  const admin = isAdmin(user);
  if (collapsed) {
    return (
      <span
        className={`inline-block w-2 h-2 rounded-full ${admin ? 'bg-amber-400' : 'bg-slate-400'}`}
        title={user.role}
      />
    );
  }
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
        admin
          ? 'bg-amber-500/15 text-amber-300 border border-amber-600/40'
          : 'bg-slate-700 text-slate-300 border border-slate-600'
      }`}
    >
      {admin ? '👑 Admin' : '🔍 Viewer'}
    </span>
  );
}

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const { user } = useAuth();

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
      <div className="flex items-center justify-between px-4 py-4 border-b border-slate-700">
        {!collapsed ? (
          <div>
            {/* Typographic title: GIC (heavy) · Plan-to-Perform (accent) · subtitle */}
            <div className="flex items-baseline gap-1.5 leading-none">
              <span className="text-xl font-black text-white tracking-tighter">GIC</span>
              <span className="text-[10px] font-bold text-blue-400 tracking-wider uppercase leading-none">
                Plan-to-Perform
              </span>
            </div>
            <div className="text-[9px] font-medium text-slate-500 tracking-widest uppercase mt-1 leading-none">
              Financial Intelligence
            </div>
            <div className="mt-2">
              <RoleBadge user={user} collapsed={false} />
            </div>
          </div>
        ) : (
          <div className="w-full flex flex-col items-center gap-1">
            <span className="text-lg font-black text-white leading-none">G</span>
            <span className="text-[8px] font-bold text-blue-400 uppercase tracking-wider">P2P</span>
            <RoleBadge user={user} collapsed />
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-slate-400 hover:text-white transition-colors ml-2"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            {collapsed ? <path d="M6 12l4-4-4-4v8z" /> : <path d="M10 12l-4-4 4-4v8z" />}
          </svg>
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-4 overflow-y-auto scrollbar-thin">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title}>
            {!collapsed && (
              <div className="px-3 mb-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                {section.title}
              </div>
            )}
            <div className="space-y-1">
              {section.items.map((item) => {
                const locked = item.permission && !can(user, item.permission);
                if (locked) {
                  return (
                    <div
                      key={item.path}
                      title={collapsed ? `${item.label} (restricted)` : 'Restricted'}
                      className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-600 cursor-not-allowed select-none"
                    >
                      <span className="text-base flex-shrink-0 opacity-60">{item.emoji}</span>
                      {!collapsed && (
                        <span className="truncate flex-1 flex items-center justify-between">
                          {item.label}
                          <span aria-hidden className="text-xs">🔒</span>
                        </span>
                      )}
                    </div>
                  );
                }
                return (
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
                    {!collapsed && <span className="truncate">{item.label}</span>}
                  </NavLink>
                );
              })}
            </div>
          </div>
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
