"""Convert a grid (downscaled rainfall, or any raster on the CHIRPS/aligned grid) into per-panchayat values.

Per the project guide section 5b: load boundary polygons, confirm raster and vector share a CRS (never assume -
see check_crs_match.py), compute zonal statistics (mean, P10, P90, max) of the raster inside each polygon.

Output: one row per village per day, with mean/P10/P50/P90/max rainfall - the actual per-panchayat product this
whole pipeline exists to produce.
"""
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely.geometry
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry

BOUNDARIES_PATH = Path(__file__).resolve().parents[2] / "data" / "boundaries" / "pune_villages.geojson"
TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"

FINE_GRID_SHAPE = (35, 50)

# Per-village pixel masks in-process cache, keyed by boundaries path - see _get_village_masks()'s own
# docstring for why this exists (rasterizing 2003 real polygons takes ~7-27s, and compute_village_stats
# used to pay that cost on every single call).
_mask_cache: dict[str, tuple[gpd.GeoDataFrame, np.ndarray]] = {}

# The flat pixel-to-village incidence arrays compute_village_stats() vectorizes over - geometry-derived,
# so cached once alongside the masks (see _get_flat_pixel_index()'s own docstring for why this exists).
_flat_index_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}


def get_chirps_grid_transform() -> tuple[Affine, str]:
    """Same grid the training pairs were built on (see build_training_pairs.py / resample_to_chirps_grid.py):
    CHIRPS-derived, 0.05deg cells, trimmed to 35x50, EPSG:4326.

    IMPORTANT: build_training_pairs.py trims the original 38-row fine grid to 35 rows by keeping the FIRST 35
    (southernmost) and dropping the last 3 (northernmost) - see that script's docstring for why that edge is
    safe to trim. The north edge of the TRIMMED grid is therefore NOT the original array's northernmost
    latitude - using the untrimmed lat[-1] here silently shifts the whole raster's assumed position by 3
    pixels (0.15deg) north of where the data actually is, misaligning every zonal-stats sample. Must use the
    trimmed array's own north edge.
    """
    import xarray as xr
    sample = Path(__file__).resolve().parents[2] / "data" / "raw" / "chirps" / "chirps_pune_2023_07.nc"
    ds = xr.open_dataset(sample)
    lat = ds.latitude.values
    lon = ds.longitude.values
    ds.close()

    trimmed_rows = (len(lat) // 5) * 5  # matches build_training_pairs.py's COARSEN_FACTOR trim
    lat_trimmed = lat[:trimmed_rows]

    res_lat = abs(lat[1] - lat[0])
    res_lon = abs(lon[1] - lon[0])
    west = lon[0] - res_lon / 2
    north = lat_trimmed[-1] + res_lat / 2  # north edge of the TRIMMED grid, not the original
    transform = rasterio.transform.from_origin(west, north, res_lon, res_lat)
    return transform, "EPSG:4326"


def get_cached_villages() -> gpd.GeoDataFrame:
    """The same villages GeoDataFrame _get_village_masks() already reads and caches - a public accessor
    so other modules don't pay for their own separate gpd.read_file(BOUNDARIES_PATH) call (real bug
    found via profiling: backend/villages.py's bulk_village_stats() was doing exactly that on every
    single /forecast/map request - re-parsing the whole 2,003-polygon GeoJSON from disk, ~1.8-2.3s
    wasted per call for a file that never changes at runtime)."""
    villages, _ = _get_village_masks()
    return villages


def _get_village_masks() -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Rasterizes every village polygon into a boolean pixel mask against the shared fine grid, once per
    process, then caches it.

    This existed as a single rasterstats.zonal_stats() call until real timing (not just correctness)
    caught a severe performance bug: rasterizing 2003 real polygons takes ~15-27s, and that whole cost
    was being paid again on every single call to compute_village_stats() - and backend/villages.py calls
    it up to 4 times per API request (block/p10/p50/p90), so a single /forecast/map request could take
    6+ minutes wall-clock (confirmed via direct curl timing during frontend integration testing). The
    polygon geometry never changes between calls, only the raster values do, so the mask only needs to
    be built once and reused - verified to produce numerically identical results to the old
    rasterstats-based path (mean within 1e-5 mm from float summation order, p50/max exact) on real data
    before this replaced it.
    """
    key = str(BOUNDARIES_PATH)
    if key in _mask_cache:
        return _mask_cache[key]

    villages = gpd.read_file(BOUNDARIES_PATH)
    transform, crs = get_chirps_grid_transform()
    assert str(villages.crs) == crs, (
        f"CRS mismatch: villages are {villages.crs}, raster grid is {crs} - run check_crs_match.py, do not assume"
    )

    # all_touched=True (not the default False): village polygons are often smaller than a single ~5.5km
    # CHIRPS cell, so "pixel centre must fall inside the polygon" left 1487/2003 villages (74%) with zero
    # coverage in testing. all_touched=True (any pixel the polygon overlaps at all) gives 100% instead.
    masks = np.zeros((len(villages), *FINE_GRID_SHAPE), dtype=bool)
    for i, geom in enumerate(villages.geometry):
        masks[i] = rasterio.features.geometry_mask(
            [geom], out_shape=FINE_GRID_SHAPE, transform=transform, all_touched=True, invert=True
        )

    _mask_cache[key] = (villages, masks)
    return villages, masks


def _get_flat_pixel_index() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Precomputes, once, a vectorized "which pixels belong to which village" index from the boolean
    masks - the geometry never changes between calls, only the raster values do, exactly the same
    reasoning as _get_village_masks() itself.

    Real timing caught a second performance bug on top of the mask-rasterization one: even with masks
    cached, compute_village_stats()'s Python-level for loop over 2003 villages - each calling
    np.percentile() three separate times (p10/p50/p90) - took 6-20s per call (confirmed via direct
    profiling; backend/villages.py's bulk_village_stats() calls it 4x per /forecast/map request, so a
    single date change could take 25-80s wall-clock). Each individual numpy call on a tiny 1-11 pixel
    array is cheap, but ~24,000+ of them (2003 villages x 4 rasters x 3 percentiles) pay real per-call
    dispatch overhead that adds up. This index lets compute_village_stats() replace that whole loop with
    a handful of single vectorized numpy calls across all villages at once instead.

    Returns (village_ids, pos_in_row, pixel_positions, pixel_counts, max_k): for every True pixel across
    every mask, which village it belongs to (village_ids) and its position within that village's own run
    (pos_in_row, 0..pixel_counts[village]-1) - together these scatter each pixel into a
    (n_villages, max_k) padded array in one fancy-indexing assignment. pixel_positions indexes into the
    flattened (35*50,) raster. max_k is the largest per-village pixel count, i.e. the padded array's
    needed width."""
    key = str(BOUNDARIES_PATH)
    if key in _flat_index_cache:
        return _flat_index_cache[key]

    _, masks = _get_village_masks()
    n = masks.shape[0]
    flat_masks = masks.reshape(n, -1)
    pixel_counts = flat_masks.sum(axis=1)
    # np.nonzero on a 2D array scans row-major - village_ids comes out already sorted by village, and
    # within each village's own run, pixel_positions comes out already sorted ascending.
    village_ids, pixel_positions = np.nonzero(flat_masks)
    starts = np.repeat(np.cumsum(pixel_counts) - pixel_counts, pixel_counts)
    pos_in_row = np.arange(len(village_ids)) - starts
    max_k = int(pixel_counts.max()) if len(pixel_counts) else 0

    result = (village_ids, pos_in_row, pixel_positions, pixel_counts, max_k)
    _flat_index_cache[key] = result
    return result


def compute_village_stats(raster_2d: np.ndarray, date_label: str) -> pd.DataFrame:
    """raster_2d: a single day's rainfall grid, shape must match the trimmed CHIRPS grid (35, 50), ascending
    row order flipped to match rasterio's north-up convention (row 0 = northernmost).

    Vectorized across all villages at once (see _get_flat_pixel_index()'s docstring for why) - same exact
    statistics (nanmean/nanpercentile/nanmax over each village's real, non-NaN pixels) as the original
    per-village loop, verified numerically identical on real data before replacing it."""
    villages, _ = _get_village_masks()

    # rasterio expects row 0 = north; our stored arrays have row 0 = south (ascending lat) after the
    # earlier lat_ascending flip in resample_to_chirps_grid.py / build_training_pairs.py - flip here to match.
    raster_north_up = raster_2d[::-1, :]
    flat_raster = raster_north_up.reshape(-1)

    n = len(villages)
    village_ids, pos_in_row, pixel_positions, _, max_k = _get_flat_pixel_index()

    if max_k == 0:
        mean_mm = p10_mm = p50_mm = p90_mm = max_mm = np.full(n, np.nan)
        pixel_count = np.zeros(n, dtype=int)
    else:
        # NaN-padded (n_villages, max_k) grid: row i holds village i's real pixel values in the first
        # pixel_counts[i] slots, NaN after that - nanmean/nanpercentile/nanmax ignore both the padding
        # NaNs and any genuine no-data NaNs the same way the original per-pixel filter did.
        padded = np.full((n, max_k), np.nan)
        padded[village_ids, pos_in_row] = flat_raster[pixel_positions]

        with warnings.catch_warnings():
            # A village with zero real coverage (padded row all-NaN) makes nanmean/nanpercentile/nanmax
            # warn "Mean/All-NaN slice encountered" - expected and already handled (result is NaN, same
            # as the original loop's untouched np.full(n, np.nan) default for that row).
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_mm = np.nanmean(padded, axis=1)
            p10_mm, p50_mm, p90_mm = np.nanpercentile(padded, [10, 50, 90], axis=1)
            max_mm = np.nanmax(padded, axis=1)
        pixel_count = np.sum(~np.isnan(padded), axis=1).astype(int)

    result = villages[["NAME", "SUB_DIST", "CEN_2001"]].copy()
    result["date"] = date_label
    result["mean_mm"] = mean_mm
    result["p10_mm"] = p10_mm
    result["p50_mm"] = p50_mm
    result["p90_mm"] = p90_mm
    result["max_mm"] = max_mm
    result["pixel_count"] = pixel_count
    return result


def village_stats_by_index(raster_2d: np.ndarray, row_index: int) -> dict:
    """Same stats as one row of compute_village_stats(), but for exactly one village - O(pixels in that
    village's own mask, typically 1-11) instead of O(all 2003 villages).

    compute_village_stats() loops over every village to build the full per-day table, which is the
    right tool for the map endpoint (it genuinely needs all of them) but was also being reused for
    single-village lookups (backend/villages.py's panchayat_value(), called 4x per /forecast request
    and 3x more for the sparkline) - looping over all 2003 villages 12 times just to keep 1 row each
    time. row_index must be a real positional index into the same villages GeoDataFrame
    _get_village_masks() returns (see VillageIndex.find() in backend/villages.py, which now returns one)."""
    _, masks = _get_village_masks()
    raster_north_up = raster_2d[::-1, :]
    pixels = raster_north_up[masks[row_index]]
    pixels = pixels[~np.isnan(pixels)]
    if pixels.size == 0:
        return {"mean_mm": None, "p10_mm": None, "p50_mm": None, "p90_mm": None, "max_mm": None, "pixel_count": 0}
    p10, p50, p90 = np.percentile(pixels, [10, 50, 90])  # one sorted pass instead of three
    return {
        "mean_mm": float(pixels.mean()),
        "p10_mm": float(p10),
        "p50_mm": float(p50),
        "p90_mm": float(p90),
        "max_mm": float(pixels.max()),
        "pixel_count": int(pixels.size),
    }


COARSEN_FACTOR = 5  # matches backend/inference.py's own ch, cw = h // 5, w // 5 (block_value construction)

_district_union_cache: dict[str, BaseGeometry] = {}


def _get_district_union():
    """The real Pune district's outer silhouette (union of all 2,003 village polygons), cached like the
    village masks - block_grid_geojson() clips every grid rectangle to this so the block layer's outer
    edge matches the exact same boundary the panchayat layer already uses, instead of the CHIRPS grid's
    own raw rectangular extent (which is padded well beyond the real district on every side - confirmed
    visually, see DEV_LOG: raw grid squares covered non-Pune land at the corners and left transparent
    gaps in the panchayat pane wherever a village polygon didn't reach the raw grid's edge)."""
    key = str(BOUNDARIES_PATH)
    if key not in _district_union_cache:
        villages, _ = _get_village_masks()
        union = villages.union_all() if hasattr(villages, "union_all") else villages.unary_union
        _district_union_cache[key] = union
    return _district_union_cache[key]


_grid_cell_geometry_cache: dict[str, list[dict]] = {}


def _get_block_grid_cells() -> list[dict]:
    """The block grid's real geometry - each candidate 0.25deg cell clipped to the district silhouette,
    computed once and cached, since the shape is fixed and only block_mm (a single float per cell)
    changes per date.

    This existed inline inside block_grid_geojson() until profiling caught it repeating the same ~70
    shapely intersections (a rectangle against the whole 2,003-polygon district union) on every single
    date change even though the result is identical every time - confirmed via direct timing: ~0.8-1s
    wasted per /forecast/map call for geometry that never changes. Splitting geometry (cached, computed
    once) from value (looked up fresh per date) turns that into a one-time cost."""
    key = str(BOUNDARIES_PATH)
    if key in _grid_cell_geometry_cache:
        return _grid_cell_geometry_cache[key]

    transform, _ = get_chirps_grid_transform()
    district_union = _get_district_union()
    n_rows, n_cols = FINE_GRID_SHAPE[0] // COARSEN_FACTOR, FINE_GRID_SHAPE[1] // COARSEN_FACTOR

    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            west, north = transform * (c * COARSEN_FACTOR, r * COARSEN_FACTOR)
            east, south = transform * ((c + 1) * COARSEN_FACTOR, (r + 1) * COARSEN_FACTOR)
            cell = shapely.geometry.box(west, south, east, north)
            clipped = cell.intersection(district_union)
            if clipped.is_empty:
                continue
            # (row, col): the one fine-grid pixel each cell's block_value is sampled from (constant
            # across the whole cell by construction, see backend/inference.py's block_value).
            cells.append({
                "id": r * n_cols + c,
                "row": r * COARSEN_FACTOR,
                "col": c * COARSEN_FACTOR,
                "geometry": clipped.__geo_interface__,
            })

    _grid_cell_geometry_cache[key] = cells
    return cells


def block_grid_geojson(block_value: np.ndarray) -> dict:
    """The real 0.25deg CHIRPS block grid as GeoJSON rectangles, each carrying the constant block_mm
    value backend/inference.py already computed for that cell (predict_day()'s block_value, (35, 50)
    south-ascending like every other raster this module handles) - clipped to the real district
    silhouette, so a cell that only partially overlaps the district is trimmed to the true boundary
    instead of painting neighbouring non-Pune land, and a cell with no overlap at all is dropped entirely.

    Exists so the dashboard can render the coarse block layer as actual grid squares instead of the
    same 2,003 village-shaped patches used for the panchayat layer sharing a flat colour per block -
    real data either way, but village-shaped patches read as noisy blotches instead of the genuinely
    blocky ~25km cells the comparison is supposed to show (see DEV_LOG)."""
    cells = _get_block_grid_cells()
    raster_north_up = block_value[::-1, :]

    features = []
    for cell in cells:
        value = float(raster_north_up[cell["row"], cell["col"]])
        features.append({
            "type": "Feature",
            "id": cell["id"],
            "geometry": cell["geometry"],
            "properties": {"block_mm": None if np.isnan(value) else round(value, 2)},
        })
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    dates = np.load(TRAINING_PAIRS_DIR / "dates.npy")

    # Demo: run on a single known day rather than all 5,582 (that belongs in a batch job once this is
    # validated) - July 15 2023, the same day already cross-validated between CHIRPS and IMD.
    idx = np.where(dates == "2023-07-15")[0]
    if len(idx) == 0:
        raise ValueError("2023-07-15 not found in dates.npy")
    idx = idx[0]

    day_grid = fine[idx]
    result = compute_village_stats(day_grid, dates[idx])

    print(f"Computed zonal stats for {len(result)} villages on {dates[idx]}")
    print(result[["NAME", "SUB_DIST", "mean_mm", "p10_mm", "p90_mm"]].describe())
    print()
    print("Sample rows:")
    print(result[["NAME", "SUB_DIST", "mean_mm", "p10_mm", "p50_mm", "p90_mm", "max_mm"]].head(10))

    top5 = result.nlargest(5, "mean_mm")
    print("\nTop 5 wettest villages this day:")
    print(top5[["NAME", "SUB_DIST", "mean_mm"]])

    out_path = TRAINING_PAIRS_DIR / f"zonal_stats_{dates[idx]}.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
