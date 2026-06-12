import React, { useEffect, useRef, useState } from 'react';
import { useRealtimeContext } from '../../context/RealtimeContext';

// Format a price by magnitude/unit for compact display.
function fmtPrice(price, unit) {
  if (price === null || price === undefined) return '—';
  if (unit && unit.includes('bbl')) return price.toFixed(2);
  if (price >= 1000) return price.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return price.toFixed(2);
}

function ChangeChip({ change }) {
  const up = (change ?? 0) >= 0;
  const color = Math.abs(change ?? 0) < 0.005 ? '#94a3b8' : up ? '#34d399' : '#f87171';
  const arrow = Math.abs(change ?? 0) < 0.005 ? '▪' : up ? '▲' : '▼';
  return (
    <span className="font-mono text-xs font-semibold tabular-nums" style={{ color }}>
      {arrow} {Math.abs(change ?? 0).toFixed(2)}%
    </span>
  );
}

// A single ticker cell that flashes when its value changes.
function TickerItem({ label, value, change, unit }) {
  const prevRef = useRef(value);
  const [flash, setFlash] = useState(null);

  useEffect(() => {
    if (prevRef.current !== value && prevRef.current !== undefined) {
      const up = value >= prevRef.current;
      setFlash(up ? 'up' : 'down');
      const t = setTimeout(() => setFlash(null), 600);
      prevRef.current = value;
      return () => clearTimeout(t);
    }
    prevRef.current = value;
  }, [value]);

  const flashBg =
    flash === 'up'
      ? 'rgba(52,211,153,0.16)'
      : flash === 'down'
      ? 'rgba(248,113,113,0.16)'
      : 'transparent';

  return (
    <div
      className="flex items-center gap-2 px-4 py-2 border-r border-slate-700/70 whitespace-nowrap transition-colors duration-300"
      style={{ backgroundColor: flashBg }}
    >
      <span className="text-xs font-semibold text-slate-300 uppercase tracking-wide">{label}</span>
      <span className="font-mono text-sm font-bold text-white tabular-nums">
        {fmtPrice(value, unit)}
      </span>
      <ChangeChip change={change} />
    </div>
  );
}

export default function LiveMarketTape() {
  const { snapshot, connected, source } = useRealtimeContext();
  const s = snapshot;

  const live = source === 'live' && connected;
  const statusDot = live ? '●' : '◐';
  const statusColor = live ? '#34d399' : '#fbbf24';
  const statusLabel = live ? 'LIVE' : 'SIMULATED';

  const commodities = s?.top_commodities?.slice(0, 6) || [];
  const fx = s?.fx?.slice(0, 3) || [];

  return (
    <div
      className="rounded-xl border border-slate-700 overflow-hidden"
      style={{ backgroundColor: '#1e293b' }}
    >
      <div className="flex items-stretch">
        {/* Status block */}
        <div
          className="flex items-center gap-2 px-4 py-2 border-r border-slate-700 flex-shrink-0"
          style={{ backgroundColor: '#0f172a' }}
        >
          <span className="text-sm animate-pulse" style={{ color: statusColor }}>
            {statusDot}
          </span>
          <span className="text-xs font-bold tracking-widest" style={{ color: statusColor }}>
            {statusLabel}
          </span>
        </div>

        {/* Scrolling-feel in-place ticker */}
        <div className="flex items-stretch overflow-x-auto no-scrollbar flex-1">
          {commodities.map((c) => (
            <TickerItem
              key={c.name}
              label={c.name}
              value={c.price}
              change={c.change_pct}
              unit={c.unit}
            />
          ))}
          {fx.map((f) => (
            <TickerItem key={f.pair} label={f.pair} value={f.rate} change={f.change_pct} />
          ))}
          {commodities.length === 0 && (
            <div className="px-4 py-2 text-xs text-slate-500">Awaiting market data…</div>
          )}
        </div>

        {/* Timestamp */}
        {s?.timestamp && (
          <div className="hidden md:flex items-center px-4 py-2 border-l border-slate-700 flex-shrink-0">
            <span className="text-[10px] font-mono text-slate-500">
              {new Date(s.timestamp).toLocaleTimeString()}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
