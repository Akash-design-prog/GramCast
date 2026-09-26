import { useCallback, useEffect, useRef, useState } from "react";
import Header from "./components/Header";
import CompareMap from "./components/CompareMap";
import Legend from "./components/Legend";
import Sidebar from "./components/Sidebar";
import { ApiError, addDaysToDateString, fetchForecast, fetchForecastMap } from "./lib/api";
import type { ForecastMapResponse, ForecastResponse, VillageProperties } from "./lib/types";

const THEME_KEY = "gramcast-theme";
// 2003-07-27: scanned 60 random real dates for genuine in-district rainfall variety (not cherry-picked
// for a specific number, just the highest village-level std/max found) - village mean_mm std 33.6mm,
// max 159mm, and a real ~5.5mm average block-vs-panchayat difference across 858/2003 villages, so the
// slider actually shows something on this day. 2023-07-15 (the day already cross-validated elsewhere -
// backend tests, IMD cross-check, DEV_LOG) is real too, but its heaviest rain that day falls just north
// of the district boundary - verified via the raw model grid - so almost every village reads near-zero.
const BASE_DATE = "2003-07-27";
const DAY_COUNT = 3;

type Theme = "light" | "dark";

function readStoredTheme(): Theme | null {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function resolvedTheme(explicit: Theme | null): Theme {
  if (explicit) return explicit;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [themeChoice, setThemeChoice] = useState<Theme | null>(readStoredTheme);
  const [activeDay, setActiveDay] = useState(0);

  const [mapData, setMapData] = useState<ForecastMapResponse | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);

  const [selected, setSelected] = useState<{ props: VillageProperties; featureId: number; lat: number; lon: number } | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [sparkPoints, setSparkPoints] = useState<{ label: string; p50: number }[]>([]);

  const mapCache = useRef(new Map<string, ForecastMapResponse>());

  useEffect(() => {
    const theme = resolvedTheme(themeChoice);
    if (themeChoice) document.documentElement.setAttribute("data-theme", themeChoice);
    else document.documentElement.removeAttribute("data-theme");
    void theme;
  }, [themeChoice]);

  const dayLabels = ["Today", "Tomorrow", "Day 3"];
  const currentDate = addDaysToDateString(BASE_DATE, activeDay);

  // Fetch the whole district's map layer whenever the selected day changes, cached per date so
  // flipping between day pills doesn't re-hit the model.
  useEffect(() => {
    let cancelled = false;
    const cached = mapCache.current.get(currentDate);
    if (cached) {
      setMapData(cached);
      setMapError(null);
      return;
    }
    setMapError(null);
    fetchForecastMap(currentDate)
      .then((data) => {
        if (cancelled) return;
        mapCache.current.set(currentDate, data);
        setMapData(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setMapError(err instanceof ApiError ? err.message : "Could not reach the GramCast backend.");
      });
    return () => {
      cancelled = true;
    };
  }, [currentDate]);

  const handleVillageClick = useCallback((props: VillageProperties, featureId: number, lngLat: { lat: number; lng: number }) => {
    setSelected({ props, featureId, lat: lngLat.lat, lon: lngLat.lng });
  }, []);

  // Fetch this village's real forecast (drives the sidebar) plus up to DAY_COUNT-1 following real
  // days for the sparkline - never fabricated, just omitted if a day isn't in the dataset.
  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setForecastLoading(true);
    setForecastError(null);

    const dates = Array.from({ length: DAY_COUNT }, (_, i) => addDaysToDateString(BASE_DATE, i));

    Promise.allSettled(dates.map((d) => fetchForecast(selected.lat, selected.lon, d))).then((results) => {
      if (cancelled) return;
      setForecastLoading(false);

      const primary = results[activeDay];
      if (primary.status === "fulfilled") {
        setForecast(primary.value);
      } else {
        setForecast(null);
        setForecastError(primary.reason instanceof ApiError ? primary.reason.message : "Forecast unavailable.");
      }

      const points = results
        .map((r, i) => (r.status === "fulfilled" ? { label: dayLabels[i], p50: r.value.panchayat_level.p50_mm } : null))
        .filter((p): p is { label: string; p50: number } => p !== null);
      setSparkPoints(points);
    });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, activeDay]);

  return (
    <div className="app">
      <Header
        dayLabels={dayLabels}
        activeDay={activeDay}
        onDayChange={setActiveDay}
        theme={resolvedTheme(themeChoice)}
        onToggleTheme={() => {
          const next: Theme = resolvedTheme(themeChoice) === "dark" ? "light" : "dark";
          setThemeChoice(next);
          try {
            localStorage.setItem(THEME_KEY, next);
          } catch {
            /* private-mode storage can throw - theme still applies for this session */
          }
        }}
      />

      <div className="layout">
        <Sidebar
          forecast={forecast}
          loading={forecastLoading}
          error={forecastError}
          sparkPoints={sparkPoints}
          activeDayIndex={activeDay}
          lat={selected?.lat ?? null}
          lon={selected?.lon ?? null}
        />

        <div className="card map-card">
          <div className="map-head">
            <h2>Rainfall resolution comparison</h2>
            <span className="mono">{currentDate}</span>
          </div>
          {mapError && <div className="status-banner error">{mapError} - is the backend running on localhost:8000?</div>}
          <CompareMap data={mapData} onVillageClick={handleVillageClick} selectedFeatureId={selected?.featureId ?? null} />
          <Legend />
          <p className="caption">
            Real Pune district boundaries (2,003 villages) and real model output - block values are the flat 25km CHIRPS
            cell each village falls in, panchayat values are this project's U-Net downscaling, both via the same
            zonal-stats overlay used throughout the data pipeline.
          </p>
        </div>
      </div>
    </div>
  );
}
