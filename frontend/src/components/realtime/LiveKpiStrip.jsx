import React, { useEffect, useRef, useState } from 'react';
import { useRealtimeContext } from '../../context/RealtimeContext';

function bandColor(band) {
  const b = (band || '').toLowerCase();
  if (b === 'low') return '#34d399';
  if (b === 'moderate' || b === 'elevated') return '#fbbf24';
  if (b === 'high') return '#fb923c';
  if (b === 'critical') return '#f87171';
  return '#60a5fa';
}

// A live tile that briefly flashes when the underlying value changes.
function LiveTile({ title, value, sub, subColor, accent, format }) {
  const prevRef = useRef(value);
  const [flash, setFlash] = useState(false);

  useEffect(() => {
    if (prevRef.current !== value && prevRef.current !== undefined) {
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 500);
      prevRef.current = value;
      return () => clearTimeout(t);
    }
    prevRef.current = value;
  }, [value]);

  return (
    <div
      className="rounded-xl p-4 border flex flex-col gap-1 transition-all duration-300"
      style={{
        backgroundColor: '#1e293b',
        borderColor: flash ? accent : '#334155',
        boxShadow: flash ? `0 0 0 1px ${accent}` : 'none',
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wide">
          {title}
        </span>
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{ backgroundColor: accent, opacity: flash ? 1 : 0.5 }}
        />
      </div>
      <div className="text-2xl font-bold text-white tabular-nums leading-tight">{format(value)}</div>
      {sub && (
        <span className="text-xs font-semibold tabular-nums" style={{ color: subColor || '#94a3b8' }}>
          {sub}
        </span>
      )}
    </div>
  );
}

export default function LiveKpiStrip() {
  const { snapshot } = useRealtimeContext();
  const s = snapshot || {};

  const idxChange = s.commodity_index_change_pct ?? 0;
  const idxUp = idxChange >= 0;
  const risk = s.risk_score ?? 0;
  const rColor = bandColor(s.risk_band);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <LiveTile
        title="Commodity Index"
        value={s.commodity_index ?? 100}
        accent="#f59e0b"
        format={(v) => (typeof v === 'number' ? v.toFixed(1) : '—')}
        sub={`${idxUp ? '▲' : '▼'} ${Math.abs(idxChange).toFixed(2)}% vs base`}
        subColor={idxUp ? '#34d399' : '#f87171'}
      />
      <LiveTile
        title="EBIT Nowcast"
        value={s.ebit_nowcast_gbp ?? 0}
        accent="#3b82f6"
        format={(v) => `£${(v / 1e9).toFixed(2)}bn`}
        sub="Real-time estimate"
        subColor="#94a3b8"
      />
      <LiveTile
        title="Risk Score"
        value={risk}
        accent={rColor}
        format={(v) => (typeof v === 'number' ? v.toFixed(0) : '—')}
        sub={s.risk_band || '—'}
        subColor={rColor}
      />
      <LiveTile
        title="Active Alerts"
        value={s.active_alerts ?? 0}
        accent="#f87171"
        format={(v) => String(v)}
        sub={(s.active_alerts ?? 0) > 0 ? 'Requires attention' : 'All clear'}
        subColor={(s.active_alerts ?? 0) > 0 ? '#fb923c' : '#34d399'}
      />
    </div>
  );
}
