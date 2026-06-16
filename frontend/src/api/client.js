import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const TOKEN_KEY = 'gic_token';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

// Inject Authorization header from localStorage on every request.
api.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'API error';
    const err = new Error(message);
    err.status = error.response?.status;
    return Promise.reject(err);
  }
);

// ── Client-side demo auth (used when backend is unreachable) ──────────────────
const DEMO_CREDENTIALS = { admin: 'admin123', user: 'user123' };

const MOCK_USERS = {
  mock_admin: {
    username: 'admin',
    full_name: 'Administrator',
    role: 'ADMIN',
    permissions: [
      'view_dashboard', 'view_executive_summary', 'view_forecasts', 'view_insights',
      'view_aggregated_data', 'view_market_monitor', 'view_audit_summary', 'view_warranty',
      'run_sandbox_simulation', 'run_simulation', 'edit_scenarios', 'view_raw_data',
      'manage_thresholds', 'trigger_retraining', 'trigger_data_fetch', 'view_audit_full',
      'export_reports', 'regenerate_narratives', 'manage_users',
    ],
  },
  mock_user: {
    username: 'user',
    full_name: 'Analyst',
    role: 'USER',
    permissions: [
      'view_dashboard', 'view_executive_summary', 'view_forecasts', 'view_insights',
      'view_aggregated_data', 'view_market_monitor', 'view_audit_summary', 'view_warranty',
      'run_sandbox_simulation',
    ],
  },
};

/**
 * Derive the WebSocket base URL from the configured API base.
 * http -> ws, https -> wss. Defaults to ws://localhost:8000.
 */
export const WS_BASE = (() => {
  try {
    const u = new URL(BASE_URL);
    const proto = u.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${u.host}`;
  } catch {
    return 'ws://localhost:8000';
  }
})();

export function getWsUrl(path = '/ws/market') {
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${WS_BASE}${p}`;
}

export const gicApi = {
  // Health
  health: () => api.get('/health'),
  getModels: () => api.get('/models'),

  // Auth
  login: async (username, password) => {
    try {
      return await api.post('/auth/login', { username, password });
    } catch (e) {
      if (e.status) throw e;                            // real HTTP error — propagate
      const key = `mock_${username}`;
      if (DEMO_CREDENTIALS[username] === password && MOCK_USERS[key]) {
        return { access_token: key, user: MOCK_USERS[key] };
      }
      throw new Error('Invalid username or password.');
    }
  },
  me: async () => {
    try {
      return await api.get('/auth/me');
    } catch (e) {
      if (e.status) throw e;                            // real HTTP error — propagate
      const token = localStorage.getItem(TOKEN_KEY);
      if (token && MOCK_USERS[token]) return MOCK_USERS[token];
      throw e;
    }
  },
  demoProfiles: () => api.get('/auth/demo-profiles'),
  permissions: () => api.get('/auth/permissions'),
  logout: () => {
    localStorage.removeItem(TOKEN_KEY);
  },

  // Realtime
  snapshot: () => api.get('/realtime/snapshot'),

  // Insights
  insightsFeed: () => api.get('/insights/feed'),
  varianceBridge: () => api.get('/insights/variance-bridge'),
  earlyWarning: () => api.get('/insights/early-warning'),
  warrantySummary: () => api.get('/insights/warranty/summary'),

  // Forecasts
  forecastCommodity: (commodity, horizonMonths = 12) =>
    api.post('/forecast/commodity', { commodity, horizon_months: horizonMonths }),
  forecastCommodityIndex: () => api.get('/forecast/commodity-index'),
  getCommodityHistory: (commodity, months = 36) =>
    api.get(`/forecast/commodity-history?commodity=${commodity}&months=${months}`),
  getElasticity: () => api.get('/forecast/elasticity'),

  // Simulation
  runScenario: (params) => api.post('/simulation/scenario', params),
  getPresets: () => api.get('/simulation/presets'),
  comparePresets: () => api.get('/simulation/compare-presets'),
  varianceDecomposition: () => api.get('/simulation/variance-decomposition'),
  monthlyFan: () => api.get('/simulation/monthly-fan'),

  // P&L
  getAnnualPnL: () => api.get('/pnl/annual'),

  // Market data (live indices + FX from Yahoo Finance)
  getMarketIndices: () => api.get('/realtime/market-indices'),
  getFxHistory: () => api.get('/realtime/fx-history'),
  refreshMarketData: () => api.post('/realtime/refresh'),

  // Governance — live audit trail, bias metrics and LLM narratives
  getAuditTrail: (limit = 50, eventType = null) => {
    const params = new URLSearchParams({ limit });
    if (eventType) params.append('event_type', eventType);
    return api.get(`/intelligence/audit?${params}`);
  },
  getBiasMetrics: () => api.get('/intelligence/bias'),
  getNarrative: (commodity) =>
    api.get(`/intelligence/narrative/${encodeURIComponent(commodity)}`),
  triggerPipelineRefresh: () => {
    api.post('/realtime/refresh').catch(() => {});
  },
};

export default api;
