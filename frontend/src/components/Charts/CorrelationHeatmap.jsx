import React, { useState } from 'react';

/**
 * Interactive correlation heatmap. Hovering a cell highlights it and surfaces
 * the pair + coefficient in a floating readout. Pure SVG/divs — no chart lib.
 *
 * Props:
 *  - labels: string[]            (axis labels, N)
 *  - matrix: number[][]          (N x N correlation values in [-1, 1])
 */
function corrColor(v) {
  // Diverging blue (negative) -> slate (zero) -> red (positive)
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    const a = 0.12 + t * 0.78;
    return `rgba(239,68,68,${a.toFixed(3)})`;
  }
  const a = 0.12 + Math.abs(t) * 0.78;
  return `rgba(59,130,246,${a.toFixed(3)})`;
}

export default function CorrelationHeatmap({ labels = [], matrix = [] }) {
  const [hover, setHover] = useState(null);
  const n = labels.length;

  return (
    <div>
      <div className="overflow-x-auto">
        <div className="inline-block">
          {/* Column headers */}
          <div className="flex">
            <div style={{ width: 96 }} />
            {labels.map((l) => (
              <div
                key={l}
                className="text-[10px] text-slate-400 text-center font-medium"
                style={{ width: 40, transform: 'rotate(-45deg)', transformOrigin: 'center', height: 44, lineHeight: '44px' }}
              >
                {l.length > 6 ? l.slice(0, 6) : l}
              </div>
            ))}
          </div>

          {matrix.map((row, i) => (
            <div key={labels[i]} className="flex items-center">
              <div className="text-[11px] text-slate-300 text-right pr-2 font-medium" style={{ width: 96 }}>
                {labels[i]}
              </div>
              {row.map((v, j) => {
                const active = hover && hover.i === i && hover.j === j;
                return (
                  <div
                    key={j}
                    onMouseEnter={() => setHover({ i, j, v })}
                    onMouseLeave={() => setHover(null)}
                    className="flex items-center justify-center text-[10px] font-mono cursor-default transition-all"
                    style={{
                      width: 40,
                      height: 32,
                      margin: 1,
                      borderRadius: 4,
                      backgroundColor: corrColor(v),
                      color: Math.abs(v) > 0.5 ? '#fff' : '#cbd5e1',
                      outline: active ? '2px solid #e2e8f0' : 'none',
                      transform: active ? 'scale(1.12)' : 'scale(1)',
                      zIndex: active ? 2 : 1,
                    }}
                  >
                    {v.toFixed(2)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between mt-4">
        <div className="text-xs text-slate-400 h-5">
          {hover ? (
            <span>
              <span className="text-slate-200 font-medium">{labels[hover.i]}</span>
              <span className="text-slate-500"> × </span>
              <span className="text-slate-200 font-medium">{labels[hover.j]}</span>
              <span className="text-slate-500"> · ρ = </span>
              <span className={hover.v >= 0 ? 'text-red-400' : 'text-blue-400'}>{hover.v.toFixed(2)}</span>
            </span>
          ) : (
            <span className="text-slate-500">Hover a cell to inspect the pairwise correlation</span>
          )}
        </div>
        <div className="flex items-center gap-2 text-[10px] text-slate-500">
          <span>-1</span>
          <span className="w-4 h-3 rounded-sm" style={{ backgroundColor: 'rgba(59,130,246,0.85)' }} />
          <span className="w-4 h-3 rounded-sm" style={{ backgroundColor: 'rgba(100,116,139,0.2)' }} />
          <span className="w-4 h-3 rounded-sm" style={{ backgroundColor: 'rgba(239,68,68,0.85)' }} />
          <span>+1</span>
        </div>
      </div>
      {n === 0 && <p className="text-slate-500 text-sm">No correlation data.</p>}
    </div>
  );
}
