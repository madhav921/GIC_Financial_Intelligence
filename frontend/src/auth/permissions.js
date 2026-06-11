// Permission strings mirroring the backend RBAC contract.
// USER (Analyst/Viewer) has a subset; ADMIN has all.
export const PERMISSIONS = {
  VIEW_DASHBOARD: 'view_dashboard',
  VIEW_EXECUTIVE_SUMMARY: 'view_executive_summary',
  VIEW_FORECASTS: 'view_forecasts',
  VIEW_INSIGHTS: 'view_insights',
  VIEW_AGGREGATED_DATA: 'view_aggregated_data',
  VIEW_MARKET_MONITOR: 'view_market_monitor',
  VIEW_AUDIT_SUMMARY: 'view_audit_summary',
  VIEW_WARRANTY: 'view_warranty',
  RUN_SANDBOX_SIMULATION: 'run_sandbox_simulation',
  RUN_SIMULATION: 'run_simulation',
  EDIT_SCENARIOS: 'edit_scenarios',
  VIEW_RAW_DATA: 'view_raw_data',
  MANAGE_THRESHOLDS: 'manage_thresholds',
  TRIGGER_RETRAINING: 'trigger_retraining',
  TRIGGER_DATA_FETCH: 'trigger_data_fetch',
  VIEW_AUDIT_FULL: 'view_audit_full',
  EXPORT_REPORTS: 'export_reports',
  REGENERATE_NARRATIVES: 'regenerate_narratives',
  MANAGE_USERS: 'manage_users',
};

/**
 * Returns true if the given user holds the given permission.
 */
export function can(user, perm) {
  if (!user || !perm) return false;
  return Array.isArray(user.permissions) && user.permissions.includes(perm);
}

/**
 * Returns true if the user has an admin role (or holds manage_users).
 */
export function isAdmin(user) {
  if (!user) return false;
  const role = (user.role || '').toUpperCase();
  return role === 'ADMIN' || can(user, PERMISSIONS.MANAGE_USERS);
}
