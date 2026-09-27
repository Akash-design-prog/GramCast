import { RAINFALL_LEGEND_LABELS, RAINFALL_RAMP } from "../lib/rainfallStyle";

export default function Legend() {
  return (
    <div className="legend">
      {RAINFALL_RAMP.map((color, i) => (
        <span className="sw" key={color}>
          <span className="chip" style={{ background: color }} />
          {RAINFALL_LEGEND_LABELS[i]}
        </span>
      ))}
    </div>
  );
}
