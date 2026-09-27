import type { ExpressionSpecification } from "maplibre-gl";

// IMD's own real 24-hour rainfall-intensity categories and colors - sampled directly from the legend
// icons served by IMD's live operational rainfall map (imdgeospatial.imd.gov.in/Rainfall), not
// approximated: No Rain/Very Light to Light/Moderate/Heavy/Very Heavy/Extremely Heavy, at IMD's own
// published mm boundaries (also the exact boundaries backend/advisory/rules.py's LIGHT/MODERATE/HEAVY
// thresholds are a subset of). Fixed regardless of light/dark theme, since it encodes data, not chrome.
export const RAINFALL_RAMP = ["#ffffff", "#b2df8a", "#33a02c", "#f6fa0a", "#f8b60c", "#eb1809"] as const;
export const RAINFALL_THRESHOLDS = [0.1, 15.6, 64.5, 115.6, 204.5] as const;
export const RAINFALL_LEGEND_LABELS = ["No rain", "0.1-15.5 mm", "15.6-64.4 mm", "64.5-115.5 mm", "115.6-204.4 mm", "204.5+ mm"];
export const NO_DATA_COLOR = "#9c9686";

/** MapLibre `step` expression bucketing an mm-valued property into RAINFALL_RAMP, with a distinct
 * flat color for villages with no zonal-stats coverage (property is null). */
export function rainfallFillExpression(field: string): ExpressionSpecification {
  const stepArgs: unknown[] = [RAINFALL_RAMP[0]];
  RAINFALL_THRESHOLDS.forEach((threshold, i) => {
    stepArgs.push(threshold, RAINFALL_RAMP[i + 1]);
  });
  return [
    "case",
    ["==", ["get", field], null],
    NO_DATA_COLOR,
    ["step", ["get", field], ...stepArgs] as unknown as ExpressionSpecification,
  ] as unknown as ExpressionSpecification;
}
