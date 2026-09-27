// Mirrors backend/main.py's response shapes exactly - keep in sync with backend/tests/test_main.py,
// which is the actual contract these types describe.

export interface VillageProperties {
  name: string;
  sub_district: string;
  block_mm: number | null;
  p10_mm: number | null;
  p50_mm: number | null;
  p90_mm: number | null;
}

export type VillageFeature = GeoJSON.Feature<GeoJSON.Polygon | GeoJSON.MultiPolygon, VillageProperties>;

export interface GridCellProperties {
  block_mm: number | null;
}

export interface ForecastMapResponse extends GeoJSON.FeatureCollection<GeoJSON.Polygon | GeoJSON.MultiPolygon, VillageProperties> {
  date: string;
  model: { checkpoint: string; epoch: number };
  // The real 0.25deg CHIRPS block grid (clean rectangles), separate from `features` (village-shaped
  // patches) - lets the block-layer pane render actual grid squares instead of village boundaries.
  grid_cells: GeoJSON.FeatureCollection<GeoJSON.Polygon, GridCellProperties>;
}

export interface AdvisoryOut {
  category: string;
  advisory_text: string;
  crop_stage: string | null;
  dry_spell: boolean;
  uncertain: boolean;
}

export interface AvailableDatesResponse {
  dates: string[];
  min: string;
  max: string;
}

export interface ForecastResponse {
  village: { name: string; sub_district: string; matched_by: string };
  date: string;
  elevation_m: number;
  block_level_mm: number;
  panchayat_level: { p10_mm: number; p50_mm: number; p90_mm: number };
  advisory: AdvisoryOut;
  model: { checkpoint: string; epoch: number };
}
