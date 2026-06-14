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
    return Promise.reject(new Error(message));
  }
);

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
  login: (username, password) => api.post('/auth/login', { username, password }),
  me: () => api.get('/auth/me'),
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
  getElasticity: () => api.get('/forecast/elasticity'),

  // Simulation
  runScenario: (params) => api.post('/simulation/scenario', params),
  getPresets: () => api.get('/simulation/presets'),
  comparePresets: () => api.post('/simulation/compare-presets', {}),

  // P&L
  buildPnL: (params = {}) => api.post('/pnl/build', params),
  getAnnualPnL: () => api.get('/pnl/annual'),
};

export default api;
