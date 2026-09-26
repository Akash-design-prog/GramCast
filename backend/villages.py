"""Village lookup and boundary-overlay wiring: turns a (lat, lon) into a specific panchayat, and a
(35, 50) prediction grid into that panchayat's actual value - reusing the already-verified zonal-stats
module (`data_pipeline/overlay/zonal_stats.py`) rather than reimplementing raster/vector overlay logic.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

_OVERLAY_DIR = Path(__file__).resolve().parents[1] / "data_pipeline" / "overlay"
sys.path.insert(0, str(_OVERLAY_DIR))
from zonal_stats import BOUNDARIES_PATH, compute_village_stats  # noqa: E402


class VillageIndex:
    """Loads the village boundaries once. find() does point-in-polygon first, falling back to nearest
    centroid only if the point falls outside every polygon (e.g. slightly off due to input imprecision -
    still returns a real answer rather than a hard 404 for an otherwise-reasonable coordinate)."""

    def __init__(self):
        self.villages = gpd.read_file(BOUNDARIES_PATH)
        self._sindex = self.villages.sindex

    def find(self, lat: float, lon: float) -> dict:
        point = Point(lon, lat)
        # predicate="within": does the query POINT lie within each candidate polygon - "contains" is the
        # reverse relationship (does the point contain the polygon, which a point never does) and was a
        # real bug caught here: it silently matched nothing and fell through to the fallback every time.
        candidates = self.villages.iloc[list(self._sindex.query(point, predicate="within"))]
        if len(candidates) > 0:
            row = candidates.iloc[0]
            matched_by = "within"
        else:
            # reproject to a metric CRS before measuring distance - lat/lon degrees aren't uniform
            # distance, and geopandas warns (correctly) that .distance() on a geographic CRS is unreliable
            point_gdf = gpd.GeoSeries([point], crs=self.villages.crs).to_crs(epsg=32643)  # UTM 43N, covers Pune
            distances = self.villages.geometry.to_crs(epsg=32643).distance(point_gdf.iloc[0])
            row = self.villages.loc[distances.idxmin()]
            matched_by = "nearest_centroid_fallback"
        return {"name": row["NAME"], "sub_district": row["SUB_DIST"], "matched_by": matched_by}

    def village_row_for(self, name: str, sub_district: str) -> pd.Series:
        matches = self.villages[(self.villages["NAME"] == name) & (self.villages["SUB_DIST"] == sub_district)]
        if len(matches) == 0:
            raise ValueError(f"village '{name}' ({sub_district}) not found in boundaries")
        return matches.iloc[0]


def panchayat_value(raster_2d: np.ndarray, village_name: str, sub_district: str, date_label: str) -> dict:
    """raster_2d: (35, 50) south-ascending, e.g. a p10/p50/p90 grid from GramCastInference.predict_day.
    Returns that one village's mean/p10/p50/p90/max, computed via the same verified zonal-stats overlay
    used for the data pipeline's own zonal_stats.py output."""
    stats_df = compute_village_stats(raster_2d, date_label)
    match = stats_df[(stats_df["NAME"] == village_name) & (stats_df["SUB_DIST"] == sub_district)]
    if len(match) == 0:
        raise ValueError(f"village '{village_name}' ({sub_district}) not found in zonal-stats output")
    row = match.iloc[0]
    return {
        "mean_mm": float(row["mean_mm"]),
        "p10_mm": float(row["p10_mm"]),
        "p50_mm": float(row["p50_mm"]),
        "p90_mm": float(row["p90_mm"]),
        "max_mm": float(row["max_mm"]),
    }
