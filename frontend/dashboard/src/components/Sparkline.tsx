interface Point {
  label: string;
  p50: number;
}

/** A tiny real 3-day P50 trend line - only rendered for days that actually exist in the historical
 * dataset (see App.tsx's fetch logic); never fabricated placeholder data. */
export default function Sparkline({ points, activeIndex }: { points: Point[]; activeIndex: number }) {
  if (points.length === 0) {
    return <p className="sparkline-empty">No further days available in this dataset.</p>;
  }

  const svgW = 260;
  const svgH = 64;
  const pad = 8;
  const stepX = points.length > 1 ? (svgW - pad * 2) / (points.length - 1) : 0;
  const maxV = Math.max(...points.map((p) => p.p50), 1);

  const coords = points.map((p, i) => ({
    x: pad + stepX * i,
    y: svgH - pad - (p.p50 / maxV) * (svgH - pad * 2),
  }));

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} preserveAspectRatio="none">
      <polyline
        points={coords.map((c) => `${c.x},${c.y}`).join(" ")}
        fill="none"
        stroke="var(--accent)"
        strokeWidth={2.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {coords.map((c, i) => (
        <g key={i}>
          <circle
            cx={c.x}
            cy={c.y}
            r={i === activeIndex ? 4.5 : 3}
            fill={i === activeIndex ? "var(--accent)" : "var(--surface-2)"}
            stroke="var(--accent)"
            strokeWidth={1.5}
          />
          <text x={c.x} y={svgH - 1} textAnchor="middle" fontSize={8} fill="var(--text-muted)">
            {points[i].label}
          </text>
        </g>
      ))}
    </svg>
  );
}
