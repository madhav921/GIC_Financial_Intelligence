import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'API error';
    return Promise.reject(new Error(message));
  }
);

export const gicApi = {
  // Health
  health: () => api.get('/health'),
  getModels: () => api.get('/models'),

  // Forecasts
  forecastCommodity: (commodity, horizonMonths = 12) =>
    api.post('/forecast/commodity', { commodity, horizon_months: horizonMonths }),
  forecastCommodityIndex: () => api.post('/forecast/commodity-index', {}),
  getElasticity: () => api.post('/forecast/elasticity', {}),

  // Simulation
  runScenario: (params) => api.post('/simulation/scenario', params),
  getPresets: () => api.get('/simulation/presets'),
  comparePresets: () => api.post('/simulation/compare-presets', {}),

  // P&L
  buildPnL: (params = {}) => api.post('/pnl/build', params),
  getAnnualPnL: () => api.get('/pnl/annual'),
};

export default api;
