import React from 'react';

// Risk bands → colour + label.
const BANDS = [
  { key: 'low', max: 33, color: '#22c55e', label: 'Low' },
  { key: 'elevated', max: 66, color: '#f59e0b', label: 'Elevated' },
  { key: 'high', max: 85, color: '#fb923c', label: 'High' },
  { key: 'critical', max: 100, color: '#ef4444', label: 'Critical' },
];

function bandFor(score) {
  return BANDS.find((b) => score <= b.max) || BANDS[BANDS.length - 1];
}

function polar(cx, cy, r, angleDeg) {
  const a = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

// Build an SVG arc path between two angles (degrees, 180=left .. 0=right).
function arcPath(cx, cy, r, startAngle, endAngle) {
  const start = polar(cx, cy, r, startAngle);
  const end = polar(cx, cy, r, endAngle);
  const largeArc = Math.abs(endAngle - startAngle) > 180 ? 1 : 0;
  // sweep 1 draws clockwise for our coordinate system
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

/**
 * Semicircular risk gauge (SVG). Sweeps 180° (left) → 0° (right).
 * Props: score (0-100), band ('low'|'elevated'|'high'|'critical'), size, label.
 */
export default function RiskGauge({ score = 0, band, size = 240, label = 'Risk Score' }) {
  const clamped = Math.max(0, Math.min(100, Number(score) || 0));
  const resolvedBand = band
    ? BANDS.find((b) => b.key === String(band).toLowerCase()) || bandFor(clamped)
    : bandFor(clamped);

  const w = size;
  const h = size * 0.62;
  const cx = w / 2;
  const cy = h - 8;
  const r = w / 2 - 18;
  const stroke = 14;

  // 180° (left) to 360°/0° (right). We map score 0→180deg start, 100→360deg.
  const startAngle = 180;
  const endAngle = 360;
  const valueAngle = startAngle + (clamped / 100) * (endAngle - startAngle);

  // Coloured band segments along the track.
  let segStart = 0;
  const segments = BANDS.map((b) => {
    const a0 = startAngle + (segStart / 100) * 180;
    const a1 = startAngle + (b.max / 100) * 180;
    segStart = b.max;
    return { ...b, a0, a1 };
  });

  const needle = polar(cx, cy, r, valueAngle);

  return (
    <div className="flex flex-col items-center">
      <svg width={w} height={h + 28} viewBox={`0 0 ${w} ${h + 28}`}>
        {/* Track background */}
        <path
          d={arcPath(cx, cy, r, startAngle, endAngle)}
          fill="none"
          stroke="#1e293b"
          strokeWidth={stroke + 4}
          strokeLinecap="round"
        />
        {/* Coloured band segments */}
        {segments.map((s) => (
          <path
            key={s.key}
            d={arcPath(cx, cy, r, s.a0, s.a1)}
            fill="none"
            stroke={s.color}
            strokeWidth={stroke}
            strokeLinecap="butt"
            opacity={resolvedBand.key === s.key ? 1 : 0.28}
          />
        ))}
        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={needle.x}
          y2={needle.y}
          stroke="#e2e8f0"
          strokeWidth={3}
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={6} fill="#e2e8f0" />

        {/* Centre score */}
        <text x={cx} y={cy - r * 0.42} textAnchor="middle" fontSize={size * 0.2} fontWeight="700" fill="#ffffff">
          {clamped.toFixed(0)}
        </text>
        <text x={cx} y={cy - r * 0.18} textAnchor="middle" fontSize={size * 0.06} fill="#94a3b8">
          / 100
        </text>
      </svg>

      <div className="flex flex-col items-center -mt-2">
        <span className="text-xs text-slate-400 uppercase tracking-wide">{label}</span>
        <span
          className="mt-1 px-3 py-1 rounded-full text-sm font-bold border"
          style={{
            color: resolvedBand.color,
            borderColor: resolvedBand.color,
            backgroundColor: `${resolvedBand.color}1f`,
          }}
        >
          {resolvedBand.label}
        </span>
      </div>
    </div>
  );
}
