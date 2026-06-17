import React from 'react';
import { useAuth } from './AuthContext';
import { can } from './permissions';

/**
 * Hook returning whether the current user holds a permission.
 */
export function useCan(perm) {
  const { user } = useAuth();
  return can(user, perm);
}

function LockedChip({ label = 'Admin only' }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-slate-700/60 text-slate-400 border border-slate-600">
      <span aria-hidden>🔒</span>
      {label}
    </span>
  );
}

/**
 * Renders children only when the current user holds `permission`.
 * Otherwise renders `fallback` (defaults to a subtle locked chip).
 */
export default function PermissionGate({ permission, fallback, children }) {
  const allowed = useCan(permission);
  if (allowed) return <>{children}</>;
  if (fallback !== undefined) return <>{fallback}</>;
  return <LockedChip />;
}
