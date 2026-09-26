import type { ForecastMapResponse } from "./types";

/** [west, south, east, north] over every ring of every feature - avoids pulling in a whole geometry
 * library just to fit the initial camera to the district's real extent. */
export function boundsOf(fc: ForecastMapResponse): [number, number, number, number] {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;

  for (const feature of fc.features) {
    const polygons = feature.geometry.type === "Polygon" ? [feature.geometry.coordinates] : feature.geometry.coordinates;
    for (const rings of polygons) {
      for (const ring of rings) {
        for (const [lon, lat] of ring) {
          if (lon < west) west = lon;
          if (lon > east) east = lon;
          if (lat < south) south = lat;
          if (lat > north) north = lat;
        }
      }
    }
  }
  return [west, south, east, north];
}
