import { useEffect, useRef, useState, useCallback } from 'react';
import { getWsUrl } from '../api/client';

const HEADLINES = [
  'Commodity index stabilising after midday volatility; risk band holding at Moderate.',
  'Aluminium softening on improved supply; EBIT nowcast revised upward.',
  'FX tailwind from GBP weakness adds margin support across the basket.',
  'Copper firmness flagged as a near-term watch item by the early-warning model.',
  'Calibrated forecast confidence steady at 79% across the 12-month horizon.',
  'Steel input costs easing; variance bridge favourable vs plan this week.',
];

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// Build a plausible initial snapshot for the simulator.
function seedSnapshot() {
  return {
    timestamp: new Date().toISOString(),
    commodity_index: 100,
    commodity_index_change_pct: 0,
    risk_score: 42,
    risk_band: 'Moderate',
    top_commodities: [
      { name: 'Aluminium', price: 2350, change_pct: 0, unit: 'USD/t' },
      { name: 'Copper', price: 9650, change_pct: 0, unit: 'USD/t' },
      { name: 'Steel', price: 720, change_pct: 0, unit: 'USD/t' },
      { name: 'Nickel', price: 17800, change_pct: 0, unit: 'USD/t' },
      { name: 'Zinc', price: 2680, change_pct: 0, unit: 'USD/t' },
      { name: 'Crude (Brent)', price: 82.4, change_pct: 0, unit: 'USD/bbl' },
    ],
    fx: [
      { pair: 'GBP/USD', rate: 1.272, change_pct: 0 },
      { pair: 'EUR/USD', rate: 1.084, change_pct: 0 },
      { pair: 'USD/CNY', rate: 7.21, change_pct: 0 },
    ],
    ebit_nowcast_gbp: 1.4e9,
    headline_insight: HEADLINES[0],
    active_alerts: 2,
  };
}

// Mean-reverting random-walk mutation of the snapshot.
function tick(prev, headlineIdxRef) {
  const meanRevert = (val, mean, pull, vol) =>
    val + (mean - val) * pull + (Math.random() - 0.5) * vol;

  const idx = clamp(meanRevert(prev.commodity_index, 100, 0.05, 0.6), 92, 110);
  const idxChange = ((idx - 100) / 100) * 100;

  const commodities = prev.top_commodities.map((c) => {
    const base = c.price;
    const vol = base * 0.004;
    const next = meanRevert(base, base, 0.03, vol);
    const change = ((next - base) / base) * 100 + (Math.random() - 0.5) * 0.4;
    return { ...c, price: Number(next.toFixed(2)), change_pct: Number(change.toFixed(2)) };
  });

  const fx = prev.fx.map((f) => {
    const next = meanRevert(f.rate, f.rate, 0.04, f.rate * 0.0015);
    const change = ((next - f.rate) / f.rate) * 100 + (Math.random() - 0.5) * 0.15;
    return { ...f, rate: Number(next.toFixed(4)), change_pct: Number(change.toFixed(2)) };
  });

  const risk = clamp(meanRevert(prev.risk_score, 42, 0.06, 4), 8, 92);
  const riskBand = risk < 33 ? 'Low' : risk < 66 ? 'Moderate' : 'Elevated';

  const ebit = clamp(
    meanRevert(prev.ebit_nowcast_gbp, 1.4e9, 0.05, 1.5e7),
    1.3e9,
    1.5e9
  );

  // Occasionally rotate the headline insight.
  let headline = prev.headline_insight;
  if (Math.random() < 0.12) {
    headlineIdxRef.current = (headlineIdxRef.current + 1) % HEADLINES.length;
    headline = HEADLINES[headlineIdxRef.current];
  }

  return {
    timestamp: new Date().toISOString(),
    commodity_index: Number(idx.toFixed(2)),
    commodity_index_change_pct: Number(idxChange.toFixed(2)),
    risk_score: Number(risk.toFixed(1)),
    risk_band: riskBand,
    top_commodities: commodities,
    fx,
    ebit_nowcast_gbp: Math.round(ebit),
    headline_insight: headline,
    active_alerts: prev.active_alerts,
  };
}

/**
 * useRealtime — connects to the /ws/market WebSocket and streams snapshots.
 * Falls back to a client-side simulator if the socket fails or no backend is
 * reachable, so the dashboard always ticks (e.g. on Vercel with no backend).
 *
 * @returns {{ snapshot: object|null, connected: boolean, source: 'live'|'simulated' }}
 */
export default function useRealtime() {
  const [snapshot, setSnapshot] = useState(null);
  const [connected, setConnected] = useState(false);
  const [source, setSource] = useState('simulated');

  const wsRef = useRef(null);
  const simTimerRef = useRef(null);
  const headlineIdxRef = useRef(0);
  const snapshotRef = useRef(null);
  const mountedRef = useRef(true);

  const setSnap = useCallback((s) => {
    snapshotRef.current = s;
    if (mountedRef.current) setSnapshot(s);
  }, []);

  const startSimulator = useCallback(() => {
    if (simTimerRef.current) return;
    if (!snapshotRef.current) setSnap(seedSnapshot());
    setSource('simulated');
    setConnected(false);
    simTimerRef.current = setInterval(() => {
      const base = snapshotRef.current || seedSnapshot();
      setSnap(tick(base, headlineIdxRef));
    }, 2000);
  }, [setSnap]);

  const stopSimulator = useCallback(() => {
    if (simTimerRef.current) {
      clearInterval(simTimerRef.current);
      simTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    let ws;
    let fallbackTimer;

    try {
      ws = new WebSocket(getWsUrl('/ws/market'));
      wsRef.current = ws;

      // If the socket does not open promptly, start the simulator.
      fallbackTimer = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) startSimulator();
      }, 2500);

      ws.onopen = () => {
        clearTimeout(fallbackTimer);
        stopSimulator();
        if (mountedRef.current) {
          setConnected(true);
          setSource('live');
        }
      };

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          setSnap(data);
          if (mountedRef.current) {
            setConnected(true);
            setSource('live');
          }
        } catch {
          // ignore malformed frames
        }
      };

      ws.onerror = () => {
        startSimulator();
      };

      ws.onclose = () => {
        clearTimeout(fallbackTimer);
        startSimulator();
      };
    } catch {
      startSimulator();
    }

    return () => {
      mountedRef.current = false;
      clearTimeout(fallbackTimer);
      stopSimulator();
      if (ws) {
        ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
        try {
          ws.close();
        } catch {
          // noop
        }
      }
    };
  }, []); // eslint-disable-line

  return { snapshot, connected, source };
}

export { useRealtime };
