"""Convert a grid (downscaled rainfall, or any raster on the CHIRPS/aligned grid) into per-panchayat values.

Per the project guide section 5b: load boundary polygons, confirm raster and vector share a CRS (never assume -
see check_crs_match.py), compute zonal statistics (mean, P10, P90, max) of the raster inside each polygon.

Output: one row per village per day, with mean/P10/P50/P90/max rainfall - the actual per-panchayat product this
whole pipeline exists to produce.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.transform import Affine

BOUNDARIES_PATH = Path(__file__).resolve().parents[2] / "data" / "boundaries" / "pune_villages.geojson"
TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"

FINE_GRID_SHAPE = (35, 50)

# Per-village pixel masks in-process cache, keyed by boundaries path - see _get_village_masks()'s own
# docstring for why this exists (rasterizing 2003 real polygons takes ~7-27s, and compute_village_stats
# used to pay that cost on every single call).
_mask_cache: dict[str, tuple[gpd.GeoDataFrame, np.ndarray]] = {}


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


def compute_village_stats(raster_2d: np.ndarray, date_label: str) -> pd.DataFrame:
    """raster_2d: a single day's rainfall grid, shape must match the trimmed CHIRPS grid (35, 50), ascending
    row order flipped to match rasterio's north-up convention (row 0 = northernmost)."""
    villages, masks = _get_village_masks()

    # rasterio expects row 0 = north; our stored arrays have row 0 = south (ascending lat) after the
    # earlier lat_ascending flip in resample_to_chirps_grid.py / build_training_pairs.py - flip here to match.
    raster_north_up = raster_2d[::-1, :]

    n = len(villages)
    mean_mm = np.full(n, np.nan)
    p10_mm = np.full(n, np.nan)
    p50_mm = np.full(n, np.nan)
    p90_mm = np.full(n, np.nan)
    max_mm = np.full(n, np.nan)
    pixel_count = np.zeros(n, dtype=int)

    for i in range(n):
        pixels = raster_north_up[masks[i]]
        pixels = pixels[~np.isnan(pixels)]
        if pixels.size == 0:
            continue
        mean_mm[i] = pixels.mean()
        p10_mm[i] = np.percentile(pixels, 10)
        p50_mm[i] = np.percentile(pixels, 50)
        p90_mm[i] = np.percentile(pixels, 90)
        max_mm[i] = pixels.max()
        pixel_count[i] = pixels.size

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
    return {
        "mean_mm": float(pixels.mean()),
        "p10_mm": float(np.percentile(pixels, 10)),
        "p50_mm": float(np.percentile(pixels, 50)),
        "p90_mm": float(np.percentile(pixels, 90)),
        "max_mm": float(pixels.max()),
        "pixel_count": int(pixels.size),
    }


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
