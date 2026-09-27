import Sparkline from "./Sparkline";
import { forecastVoiceUrl } from "../lib/api";
import type { ForecastResponse } from "../lib/types";
import { NO_DATA_COLOR, RAINFALL_RAMP } from "../lib/rainfallStyle";

// Indices line up with IMD's own category boundaries (rainfallStyle.ts): backend/advisory/rules.py's
// no_rain(<2.5)/light(2.5-15.5)/moderate(15.6-64.4) match RAMP[0..2] exactly; heavy(>=64.5) starts
// exactly where IMD's own "Heavy Rain" band (RAMP[3]) begins. RAMP[0] itself is pure white (IMD's map
// shows no colour at all for "no rain") - fine as a map fill, but invisible as a card border accent on
// the light theme's near-white surface, so that one case gets a neutral stand-in instead.
const CATEGORY_COLOR: Record<string, string> = {
  no_rain: NO_DATA_COLOR,
  light: RAINFALL_RAMP[1],
  moderate: RAINFALL_RAMP[2],
  heavy: RAINFALL_RAMP[3],
};

const CATEGORY_LABEL: Record<string, string> = {
  no_rain: "No rain",
  light: "Light rain",
  moderate: "Moderate rain",
  heavy: "Heavy rain",
};

interface Props {
  forecast: ForecastResponse | null;
  loading: boolean;
  error: string | null;
  sparkPoints: { label: string; p50: number }[];
  activeDayIndex: number;
  lat: number | null;
  lon: number | null;
}

export default function Sidebar({ forecast, loading, error, sparkPoints, activeDayIndex, lat, lon }: Props) {
  return (
    <aside className="sidebar">
      <div className="card panel">
        <p className="panel-label">Selected village</p>
        {!forecast && !loading && !error && (
          <p className="village-empty">Click a village on the panchayat layer to see its forecast.</p>
        )}
        {loading && !forecast && <p className="village-empty">Loading forecast...</p>}
        {error && !forecast && <p className="village-empty">{error}</p>}

        {forecast && (
          <>
            <h3 className="village-name">{forecast.village.name}</h3>
            <p className="village-sub">
              {forecast.village.sub_district} taluka, gram panchayat · {forecast.elevation_m.toFixed(0)}m elevation
              {loading && " · updating..."}
            </p>

            <div className="value-row">
              <span className="k">25km block forecast</span>
              <span className="v mono">{forecast.block_level_mm.toFixed(1)} mm</span>
            </div>
            <div className="value-row">
              <span className="k">5km panchayat forecast</span>
              <span className="v big mono">{forecast.panchayat_level.p50_mm.toFixed(1)} mm</span>
            </div>

            <div className="band">
              <div className="band-track">
                {(() => {
                  const { p10_mm, p50_mm, p90_mm } = forecast.panchayat_level;
                  const maxScale = Math.max(p90_mm, 60) * 1.05;
                  return (
                    <>
                      <div
                        className="band-fill"
                        style={{ left: `${(p10_mm / maxScale) * 100}%`, width: `${((p90_mm - p10_mm) / maxScale) * 100}%` }}
                      />
                      <div className="band-p50" style={{ left: `${(p50_mm / maxScale) * 100}%` }} />
                    </>
                  );
                })()}
              </div>
              <div className="band-labels mono">
                <span>P10 {forecast.panchayat_level.p10_mm.toFixed(0)}mm</span>
                <span>P90 {forecast.panchayat_level.p90_mm.toFixed(0)}mm</span>
              </div>
            </div>

            <div className="sparkline">
              <p className="panel-label" style={{ marginTop: 14 }}>
                Trend (days leading up to this date)
              </p>
              <Sparkline points={sparkPoints} activeIndex={activeDayIndex} />
            </div>
          </>
        )}
      </div>

      {forecast && (
        <div className="card advisory" style={{ borderLeftColor: CATEGORY_COLOR[forecast.advisory.category] ?? "var(--accent)" }}>
          <span className="tag">{CATEGORY_LABEL[forecast.advisory.category] ?? "Advisory"}</span>
          <p>{forecast.advisory.advisory_text}</p>
          <div className="flags">
            {forecast.advisory.dry_spell && <span className="flag">Dry spell</span>}
            {forecast.advisory.uncertain && <span className="flag">Uncertain band</span>}
          </div>
          {lat !== null && lon !== null && (
            <a className="voice-link" href={forecastVoiceUrl(lat, lon, forecast.date)} target="_blank" rel="noreferrer">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M11 5 6 9H2v6h4l5 4V5zM19 12a7 7 0 0 0-3-5.7M15.5 8.5a3.5 3.5 0 0 1 0 7" />
              </svg>
              Play Marathi voice note
            </a>
          )}
        </div>
      )}
    </aside>
  );
}
