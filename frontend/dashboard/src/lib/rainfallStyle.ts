import type { ExpressionSpecification } from "maplibre-gl";

// Same 6-bucket meteorological ramp used across the whole app (map fill, legend swatches, advisory
// stripe) - fixed regardless of light/dark theme, since it encodes data, not chrome.
export const RAINFALL_RAMP = ["#b9dfeb", "#8cc97f", "#e7cb4e", "#e5913f", "#d25f37", "#9e2e2a"] as const;
export const RAINFALL_THRESHOLDS = [5, 10, 20, 30, 50] as const;
export const RAINFALL_LEGEND_LABELS = ["0-5", "5-10", "10-20", "20-30", "30-50", "50+"];
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
