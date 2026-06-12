import React from 'react';
import { useAuth } from '../../auth/AuthContext';
import { can } from '../../auth/permissions';

/**
 * A button that is gated behind a permission. When the current user lacks the
 * permission it renders disabled with an "Admin only" lock affordance and a
 * tooltip explaining why, instead of crashing or silently doing nothing.
 *
 * Props:
 *  - permission: the PERMISSIONS.* string required to enable the action
 *  - onClick:    handler invoked only when allowed
 *  - children:   button label (enabled state)
 *  - lockedLabel: optional label override when locked
 *  - lockHint:   tooltip text shown on the locked state
 *  - className / color / disabled: styling + extra disable control
 */
export default function LockedButton({
  permission,
  onClick,
  children,
  lockedLabel,
  lockHint = 'Administrator access required',
  className = '',
  color = '#2563eb',
  hoverColor = '#1d4ed8',
  disabled = false,
  title,
  type = 'button',
}) {
  const { user } = useAuth();
  const allowed = can(user, permission);

  if (!allowed) {
    return (
      <button
        type={type}
        disabled
        title={lockHint}
        className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium border border-slate-600 text-slate-400 cursor-not-allowed select-none ${className}`}
        style={{ backgroundColor: 'rgba(100,116,139,0.12)' }}
      >
        <span aria-hidden>🔒</span>
        {lockedLabel || children}
        <span className="ml-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-slate-700/70 text-slate-400 border border-slate-600">
          Admin only
        </span>
      </button>
    );
  }

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50 ${className}`}
      style={{ backgroundColor: color }}
      onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.backgroundColor = hoverColor; }}
      onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = color; }}
    >
      {children}
    </button>
  );
}
