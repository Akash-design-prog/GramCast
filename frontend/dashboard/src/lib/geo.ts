import type { ForecastMapResponse } from "./types";

/** Vertex-average centroid of one village's polygon(s) - not a true geometric centroid (doesn't weight
 * by area), but that precision isn't needed here: it only has to land the map camera somewhere inside
 * the village when the user picks it from search, the same way clicking a spot on the map already does. */
export function roughCentroid(geometry: GeoJSON.Polygon | GeoJSON.MultiPolygon): [number, number] {
  const rings = geometry.type === "Polygon" ? geometry.coordinates : geometry.coordinates.flatMap((polygon) => polygon);
  let sumLon = 0;
  let sumLat = 0;
  let n = 0;
  for (const ring of rings) {
    for (const [lon, lat] of ring) {
      sumLon += lon;
      sumLat += lat;
      n++;
    }
  }
  return [sumLon / n, sumLat / n];
}

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
