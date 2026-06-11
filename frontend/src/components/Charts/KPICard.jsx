import React from 'react';

function formatValue(value, format) {
  if (value === null || value === undefined) return '—';
  switch (format) {
    case 'currency': {
      const abs = Math.abs(value);
      const sign = value < 0 ? '-' : '';
      if (abs >= 1e9) return `${sign}£${(abs / 1e9).toFixed(1)}B`;
      if (abs >= 1e6) return `${sign}£${(abs / 1e6).toFixed(1)}M`;
      if (abs >= 1e3) return `${sign}£${(abs / 1e3).toFixed(1)}K`;
      return `${sign}£${abs.toFixed(0)}`;
    }
    case 'percent':
      return `${Number(value).toFixed(1)}%`;
    case 'number': {
      const abs = Math.abs(value);
      const sign = value < 0 ? '-' : '';
      if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(1)}M`;
      if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(0)}K`;
      return `${sign}${abs.toLocaleString()}`;
    }
    default:
      return String(value);
  }
}

export default function KPICard({
  title,
  value,
  subtitle,
  change,
  changeType = 'neutral',
  format = 'number',
}) {
  const changeColor =
    changeType === 'up'
      ? '#22c55e'
      : changeType === 'down'
      ? '#ef4444'
      : '#94a3b8';

  const changeArrow =
    changeType === 'up' ? '▲' : changeType === 'down' ? '▼' : '—';

  return (
    <div
      className="rounded-xl p-5 border border-slate-700 flex flex-col gap-2"
      style={{ backgroundColor: '#1e293b' }}
    >
      <div className="text-xs font-medium text-slate-400 uppercase tracking-wide">
        {title}
      </div>
      <div className="text-3xl font-bold text-white tabular-nums">
        {formatValue(value, format)}
      </div>
      <div className="flex items-center justify-between mt-1">
        {subtitle && (
          <span className="text-xs text-slate-400">{subtitle}</span>
        )}
        {change !== undefined && change !== null && (
          <span
            className="flex items-center gap-1 text-sm font-semibold"
            style={{ color: changeColor }}
          >
            <span>{changeArrow}</span>
            <span>
              {typeof change === 'number'
                ? `${Math.abs(change).toFixed(1)}%`
                : change}
            </span>
          </span>
        )}
      </div>
    </div>
  );
}
