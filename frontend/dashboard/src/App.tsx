import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Header from "./components/Header";
import CompareMap from "./components/CompareMap";
import Legend from "./components/Legend";
import Sidebar from "./components/Sidebar";
import type { SearchEntry } from "./components/VillageSearch";
import { API_BASE, ApiError, fetchAvailableDates, fetchForecast, fetchForecastMap } from "./lib/api";
import { roughCentroid } from "./lib/geo";
import type { ForecastMapResponse, ForecastResponse, VillageProperties } from "./lib/types";

const THEME_KEY = "gramcast-theme";
// 1982-08-22: found by exhaustively scanning all 5,582 real dataset dates (not a sample) for the most
// even spread across all 6 IMD rainfall categories - this day has a real, substantial village count in
// every single band (349 no-rain / 551 light / 403 moderate / 244 heavy / 229 very heavy / 227 extreme),
// the best "evenness" score of any real day in the 46-year dataset. Earlier defaults (2003-07-27, before
// that 2023-07-15) were picked from a 60-date random sample, not a full scan - this replaces them with
// the actual best day found. Just the initial default; the date picker (Header) lets any real dataset
// date be chosen.
const DEFAULT_DATE = "1982-08-22";
const SPARK_WINDOW = 3;

function shortLabel(date: string): string {
  const d = new Date(date + "T00:00:00Z");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${d.getUTCDate()} ${months[d.getUTCMonth()]}`;
}

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
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState(DEFAULT_DATE);

  const [mapData, setMapData] = useState<ForecastMapResponse | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapLoading, setMapLoading] = useState(false);

  const [selected, setSelected] = useState<{ props: VillageProperties; featureId: number; lat: number; lon: number } | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [sparkPoints, setSparkPoints] = useState<{ label: string; p50: number }[]>([]);
  const [flyTarget, setFlyTarget] = useState<{ lat: number; lon: number; key: number } | null>(null);

  const mapCache = useRef(new Map<string, ForecastMapResponse>());

  useEffect(() => {
    const theme = resolvedTheme(themeChoice);
    if (themeChoice) document.documentElement.setAttribute("data-theme", themeChoice);
    else document.documentElement.removeAttribute("data-theme");
    void theme;
  }, [themeChoice]);

  // Fetched once: the real list of dates the model has predictions for (Jun-Sep monsoon season only,
  // 1981-2026, with real gaps) - the date picker only ever offers these, never a fabricated full range.
  useEffect(() => {
    fetchAvailableDates()
      .then((res) => {
        setAvailableDates(res.dates);
        if (!res.dates.includes(DEFAULT_DATE)) setSelectedDate(res.max);
      })
      .catch(() => {
        /* Header shows "Loading dates..." until this resolves; map/forecast fetches below will surface
         * their own errors if the backend never comes up at all. */
      });
  }, []);

  const currentDate = selectedDate;

  // Fetch the whole district's map layer whenever the selected day changes, cached per date so
  // flipping between day pills doesn't re-hit the model.
  useEffect(() => {
    let cancelled = false;
    const cached = mapCache.current.get(currentDate);
    if (cached) {
      setMapData(cached);
      setMapError(null);
      setMapLoading(false);
      return;
    }
    setMapError(null);
    // Deliberately does NOT clear mapData here - the map keeps showing the previous date's real
    // colors (rather than flashing blank) while the new date loads, with mapLoading driving a visible
    // "updating" overlay instead. Without that overlay this looked like a real bug the first time it
    // was hit: the sidebar (a fast single-village fetch) updated to the new date well before the
    // district-wide map did, so for a couple of seconds the map still showed the OLD date's colors
    // next to the new date's real (and sometimes very different, e.g. genuinely all-zero) sidebar value.
    setMapLoading(true);
    fetchForecastMap(currentDate)
      .then((data) => {
        if (cancelled) return;
        mapCache.current.set(currentDate, data);
        setMapData(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setMapError(err instanceof ApiError ? err.message : "Could not reach the GramCast backend.");
      })
      .finally(() => {
        if (!cancelled) setMapLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [currentDate]);

  const handleVillageClick = useCallback((props: VillageProperties, featureId: number, lngLat: { lat: number; lng: number }) => {
    setSelected({ props, featureId, lat: lngLat.lat, lon: lngLat.lng });
    // Same fly-to as picking a result from search - clicking a village directly on the map should zoom
    // in on it too, not just highlight it while leaving the camera wherever it happened to be.
    setFlyTarget({ lat: lngLat.lat, lon: lngLat.lng, key: Date.now() });
  }, []);

  // Rebuilt whenever the map layer's data changes (i.e. per selected date) - cheap (2,003 vertex-average
  // centroids), and every village's real name/taluka/properties are already loaded client-side with the
  // map data, so search needs no extra API call.
  const searchEntries = useMemo<SearchEntry[]>(() => {
    if (!mapData) return [];
    return mapData.features.map((f) => {
      const [lon, lat] = roughCentroid(f.geometry);
      return { id: f.id as number, name: f.properties.name, subDistrict: f.properties.sub_district, lat, lon };
    });
  }, [mapData]);

  const handleSearchSelect = useCallback(
    (entry: SearchEntry) => {
      const feature = mapData?.features.find((f) => f.id === entry.id);
      if (!feature) return;
      setSelected({ props: feature.properties, featureId: entry.id, lat: entry.lat, lon: entry.lon });
      setFlyTarget({ lat: entry.lat, lon: entry.lon, key: Date.now() });
    },
    [mapData],
  );

  // Fetch this village's real forecast (drives the sidebar) plus its trailing real days for the
  // sparkline - a genuine trend ending at the selected date, never fabricated placeholder data. The
  // dataset is monsoon-season-only, so "the day before" isn't always the prior calendar day; walking
  // back through the actual sorted available-dates list keeps every point real.
  useEffect(() => {
    if (!selected || availableDates.length === 0) return;
    let cancelled = false;
    setForecastLoading(true);
    setForecastError(null);

    const idx = availableDates.indexOf(selectedDate);
    const windowDates =
      idx === -1 ? [selectedDate] : availableDates.slice(Math.max(0, idx - (SPARK_WINDOW - 1)), idx + 1);

    Promise.allSettled(windowDates.map((d) => fetchForecast(selected.lat, selected.lon, d))).then((results) => {
      if (cancelled) return;
      setForecastLoading(false);

      const primary = results[results.length - 1];
      if (primary.status === "fulfilled") {
        setForecast(primary.value);
      } else {
        setForecast(null);
        setForecastError(primary.reason instanceof ApiError ? primary.reason.message : "Forecast unavailable.");
      }

      const points = results
        .map((r, i) =>
          r.status === "fulfilled" ? { label: shortLabel(windowDates[i]), p50: r.value.panchayat_level.p50_mm } : null,
        )
        .filter((p): p is { label: string; p50: number } => p !== null);
      setSparkPoints(points);
    });

    return () => {
      cancelled = true;
    };
  }, [selected, selectedDate, availableDates]);

  return (
    <div className="app">
      <Header
        selectedDate={selectedDate}
        onDateChange={setSelectedDate}
        availableDates={availableDates}
        searchEntries={searchEntries}
        onSearchSelect={handleSearchSelect}
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
          activeDayIndex={sparkPoints.length - 1}
          lat={selected?.lat ?? null}
          lon={selected?.lon ?? null}
        />

        <div className="card map-card">
          <div className="map-head">
            <h2>Rainfall resolution comparison</h2>
            <span className="mono">{currentDate}</span>
          </div>
          {mapError && <div className="status-banner error">{mapError} - is the backend running at {API_BASE}?</div>}
          <div className="map-stage-wrap">
            <CompareMap
              data={mapData}
              onVillageClick={handleVillageClick}
              selectedFeatureId={selected?.featureId ?? null}
              flyTarget={flyTarget}
            />
            {mapLoading && (
              <div className="map-updating-overlay">
                <span className="map-updating-spinner" />
                Updating for {currentDate}...
              </div>
            )}
          </div>
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
