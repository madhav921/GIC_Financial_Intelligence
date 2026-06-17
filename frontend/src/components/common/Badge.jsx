import React from 'react';

const COLOR_MAP = {
  green:  { bg: 'rgba(34,197,94,0.15)',  border: '#16a34a', text: '#4ade80' },
  yellow: { bg: 'rgba(245,158,11,0.15)', border: '#d97706', text: '#fbbf24' },
  red:    { bg: 'rgba(239,68,68,0.15)',  border: '#dc2626', text: '#f87171' },
  blue:   { bg: 'rgba(59,130,246,0.15)', border: '#2563eb', text: '#60a5fa' },
  slate:  { bg: 'rgba(100,116,139,0.15)',border: '#475569', text: '#94a3b8' },
};

export default function Badge({ label, color = 'blue', className = '' }) {
  const style = COLOR_MAP[color] || COLOR_MAP.blue;
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold border ${className}`}
      style={{
        backgroundColor: style.bg,
        borderColor: style.border,
        color: style.text,
      }}
    >
      {label}
    </span>
  );
}
