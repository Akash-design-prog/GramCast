import type { AvailableDatesResponse, ForecastMapResponse, ForecastResponse } from "./types";

// Points at the FastAPI backend (backend/main.py). Set VITE_API_BASE_URL in .env.local for a
// deployed backend; defaults to the local dev server started with `uvicorn backend.main:app --reload`.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
export { API_BASE };

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string, params: Record<string, string>): Promise<T> {
  const url = new URL(path, API_BASE);
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);

  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? `${path} failed with ${res.status}`);
  }
  return res.json();
}

export function fetchAvailableDates(): Promise<AvailableDatesResponse> {
  return getJson<AvailableDatesResponse>("/dates", {});
}

export function fetchForecastMap(date: string): Promise<ForecastMapResponse> {
  return getJson<ForecastMapResponse>("/forecast/map", { date });
}

export function fetchForecast(lat: number, lon: number, date: string, cropStage?: string): Promise<ForecastResponse> {
  const params: Record<string, string> = { lat: String(lat), lon: String(lon), date };
  if (cropStage) params.crop_stage = cropStage;
  return getJson<ForecastResponse>("/forecast", params);
}

export function forecastVoiceUrl(lat: number, lon: number, date: string, lang = "mr"): string {
  const url = new URL("/forecast/voice", API_BASE);
  url.searchParams.set("lat", String(lat));
  url.searchParams.set("lon", String(lon));
  url.searchParams.set("date", date);
  url.searchParams.set("lang", lang);
  return url.toString();
}
