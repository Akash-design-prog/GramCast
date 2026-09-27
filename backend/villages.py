"""Village lookup and boundary-overlay wiring: turns a (lat, lon) into a specific panchayat, and a
(35, 50) prediction grid into that panchayat's actual value - reusing the already-verified zonal-stats
module (`data_pipeline/overlay/zonal_stats.py`) rather than reimplementing raster/vector overlay logic.
"""
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

_OVERLAY_DIR = Path(__file__).resolve().parents[1] / "data_pipeline" / "overlay"
sys.path.insert(0, str(_OVERLAY_DIR))
from zonal_stats import BOUNDARIES_PATH, block_grid_geojson, compute_village_stats, get_cached_villages, village_stats_by_index  # noqa: E402


class VillageIndex:
    """Loads the village boundaries once. find() does point-in-polygon first, falling back to nearest
    centroid only if the point falls outside every polygon (e.g. slightly off due to input imprecision -
    still returns a real answer rather than a hard 404 for an otherwise-reasonable coordinate)."""

    def __init__(self):
        # reset_index(drop=True): must match _get_village_masks()' own row order exactly (same file, same
        # read call, no filtering) - row_index below is a positional index into that same cached mask
        # array, not a business key, so this alignment is load-bearing (see village_stats_by_index()).
        self.villages = gpd.read_file(BOUNDARIES_PATH).reset_index(drop=True)
        self._sindex = self.villages.sindex

    def find(self, lat: float, lon: float) -> dict:
        point = Point(lon, lat)
        # predicate="within": does the query POINT lie within each candidate polygon - "contains" is the
        # reverse relationship (does the point contain the polygon, which a point never does) and was a
        # real bug caught here: it silently matched nothing and fell through to the fallback every time.
        candidate_positions = list(self._sindex.query(point, predicate="within"))
        if len(candidate_positions) > 0:
            row_index = candidate_positions[0]
            matched_by = "within"
        else:
            # reproject to a metric CRS before measuring distance - lat/lon degrees aren't uniform
            # distance, and geopandas warns (correctly) that .distance() on a geographic CRS is unreliable
            point_gdf = gpd.GeoSeries([point], crs=self.villages.crs).to_crs(epsg=32643)  # UTM 43N, covers Pune
            distances = self.villages.geometry.to_crs(epsg=32643).distance(point_gdf.iloc[0])
            row_index = int(np.argmin(distances.values))  # positional, not distances.idxmin()'s index label
            matched_by = "nearest_centroid_fallback"
        row = self.villages.iloc[row_index]
        return {"name": row["NAME"], "sub_district": row["SUB_DIST"], "matched_by": matched_by, "row_index": row_index}

    def village_row_for(self, name: str, sub_district: str) -> pd.Series:
        matches = self.villages[(self.villages["NAME"] == name) & (self.villages["SUB_DIST"] == sub_district)]
        if len(matches) == 0:
            raise ValueError(f"village '{name}' ({sub_district}) not found in boundaries")
        return matches.iloc[0]

    def find_row_index_for(self, name: str, sub_district: str) -> int:
        """Like village_row_for(), but returns the positional row_index directly - self.villages was
        built with reset_index(drop=True) in __init__, so its pandas index already equals row order
        (0..N-1) exactly; no separate lookup needed."""
        matches = self.villages[(self.villages["NAME"] == name) & (self.villages["SUB_DIST"] == sub_district)]
        if len(matches) == 0:
            raise ValueError(f"village '{name}' ({sub_district}) not found in boundaries")
        return int(matches.index[0])

    def find_by_name_fragment(self, text: str) -> dict | None:
        """Case-insensitive, word-boundary match of `text` against every village name - used by the
        WhatsApp bot to parse a free-text message like "Baramati weather" into a real village, since a
        farmer can't tap a map. Picks the LONGEST matching name when several match (a real village name
        can itself be a raw substring of another, e.g. a 5-letter village embedded inside a 6-letter
        one), and requires a real word boundary (regex \\b) rather than a bare substring test - two real
        bugs caught here via "test the test" on ordinary sentences, not hypothetical:
          1. 45/2003 real villages have a BLANK name in the source boundaries file - `"" in text` is
             always True in Python, so an unguarded substring test matched every single message, even
             "hello there", against whichever blank-named village came first in row order.
          2. Without a word-boundary requirement, a short real village name can match as a fragment
             inside a completely unrelated English word (bare substring search, not "does this word
             appear"), producing a false match on ordinary chat text.
        Also matches against each name with a trailing official suffix like "(M Cl)"/"(CT)" stripped -
        a real farmer typing "Lonavala" would never type "Lonavala (M Cl)" (the boundaries file's exact
        official name), so matching only the literal full name made every suffixed village unreachable
        by a plausible real message.

        Returns None if no real (non-blank) village name matches as a whole word/phrase anywhere in the
        text. Does NOT disambiguate same-named villages by taluka (383/2003 villages share a name with
        another elsewhere in the district, see backend/main.py's forecast_map docstring) - a known,
        documented limitation for this MVP, not a silent bug."""
        text_lower = text.lower()
        best: tuple[int, int] | None = None  # (name_len, row_index)
        for row_index, name in enumerate(self.villages["NAME"]):
            name_stripped = name.strip()
            if not name_stripped:
                continue
            core_name = re.sub(r"\s*\([^)]*\)\s*$", "", name_stripped).strip()
            candidate = (core_name or name_stripped).lower()
            if not candidate:
                continue
            if re.search(r"\b" + re.escape(candidate) + r"\b", text_lower):
                if best is None or len(candidate) > best[0]:
                    best = (len(candidate), row_index)
        if best is None:
            return None
        row_index = best[1]
        row = self.villages.iloc[row_index]
        return {"name": row["NAME"], "sub_district": row["SUB_DIST"], "matched_by": "name_fragment", "row_index": row_index}


def bulk_village_stats(pred: dict) -> gpd.GeoDataFrame:
    """All ~2,000 villages' block value + P10/P50/P90 panchayat values for one day, as a single
    GeoDataFrame carrying geometry - powers the dashboard's map endpoint (one request for the whole
    district instead of one per village). pred is a GramCastInference.predict_day() result.

    NAME+SUB_DIST is NOT a unique key in the real boundaries file (98 duplicate pairs found - e.g.
    several "Pune (M Corp.)" polygons sharing one CEN_2001 census code, likely ward-level splits of one
    municipal corporation). A key-based merge on that pair silently explodes into a many-to-many join
    (caught here during testing: 2003 rows became 52.8 million). compute_village_stats() re-reads the
    same static GeoJSON in the same row order every call, so combine by row position instead - verified
    identical before trusting it, rather than assumed."""
    block_stats = compute_village_stats(pred["block_value"], "block")
    p10_stats = compute_village_stats(pred["p10"], "p10")
    p50_stats = compute_village_stats(pred["p50"], "p50")
    p90_stats = compute_village_stats(pred["p90"], "p90")

    for label, other in (("p10", p10_stats), ("p50", p50_stats), ("p90", p90_stats)):
        assert (other["NAME"].values == block_stats["NAME"].values).all(), (
            f"row order drifted between the block and {label} zonal-stats calls - positional merge unsafe"
        )

    # Reuses zonal_stats.py's own cached read (get_cached_villages()) instead of a second independent
    # gpd.read_file(BOUNDARIES_PATH) - profiling caught this as a real ~1.8-2.3s-per-call cost, paid on
    # every single /forecast/map request for a file that never changes at runtime.
    villages = get_cached_villages()[["NAME", "SUB_DIST", "geometry"]].reset_index(drop=True)
    assert (villages["NAME"].values == block_stats["NAME"].values).all(), (
        "boundaries file row order does not match zonal-stats output order - positional merge unsafe"
    )

    result = villages.copy()
    result["block_mm"] = block_stats["mean_mm"].values
    result["p10_mm"] = p10_stats["mean_mm"].values
    result["p50_mm"] = p50_stats["mean_mm"].values
    result["p90_mm"] = p90_stats["mean_mm"].values
    return result


def panchayat_value_by_index(raster_2d: np.ndarray, row_index: int) -> dict:
    """The fast path for a single village lookup - O(that village's own pixel count, typically 1-11)
    instead of panchayat_value()'s O(all 2003 villages). row_index comes from VillageIndex.find()'s
    own 'row_index' field. Used by the /forecast hot path (backend/main.py), which was calling
    panchayat_value() 4x per request (block/p10/p50/p90) and 3x more for the frontend's 3-day sparkline -
    up to ~24,000 whole-district stat computations to answer one village's forecast."""
    stats = village_stats_by_index(raster_2d, row_index)
    if stats["mean_mm"] is None:
        raise ValueError(f"village at row {row_index} has no pixel coverage for this raster")
    return {k: stats[k] for k in ("mean_mm", "p10_mm", "p50_mm", "p90_mm", "max_mm")}


def panchayat_value(raster_2d: np.ndarray, village_name: str, sub_district: str, date_label: str) -> dict:
    """raster_2d: (35, 50) south-ascending, e.g. a p10/p50/p90 grid from GramCastInference.predict_day.
    Returns that one village's mean/p10/p50/p90/max, computed via the same verified zonal-stats overlay
    used for the data pipeline's own zonal_stats.py output.

    O(all 2003 villages) - kept for lookups by name/sub_district (scripts, ad-hoc debugging, and NAME+
    SUB_DIST isn't even guaranteed unique, see bulk_village_stats' docstring). The API's actual per-
    request hot path uses panchayat_value_by_index() instead, which is O(1) in village count."""
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
