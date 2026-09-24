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
from rasterio.transform import Affine
from rasterstats import zonal_stats

BOUNDARIES_PATH = Path(__file__).resolve().parents[2] / "data" / "boundaries" / "pune_villages.geojson"
TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"


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


def compute_village_stats(raster_2d: np.ndarray, date_label: str) -> pd.DataFrame:
    """raster_2d: a single day's rainfall grid, shape must match the trimmed CHIRPS grid (35, 50), ascending
    row order flipped to match rasterio's north-up convention (row 0 = northernmost)."""
    villages = gpd.read_file(BOUNDARIES_PATH)
    transform, crs = get_chirps_grid_transform()

    assert str(villages.crs) == crs, (
        f"CRS mismatch: villages are {villages.crs}, raster grid is {crs} - run check_crs_match.py, do not assume"
    )

    # rasterio/rasterstats expect row 0 = north; our stored arrays have row 0 = south (ascending lat) after the
    # earlier lat_ascending flip in resample_to_chirps_grid.py / build_training_pairs.py - flip here to match.
    raster_north_up = raster_2d[::-1, :]

    # all_touched=True (not rasterstats' default of False): village polygons are often smaller than a single
    # ~5.5km CHIRPS cell, so the default "pixel centre must fall inside the polygon" rule left 1487/2003
    # villages (74%) with zero pixel coverage in testing. all_touched=True (include any pixel the polygon
    # overlaps at all, even partially) gives 100% coverage instead - confirmed by direct comparison.
    stats = zonal_stats(
        villages,
        raster_north_up,
        affine=transform,
        stats=["mean", "min", "max", "percentile_10", "percentile_50", "percentile_90", "count"],
        nodata=np.nan,
        all_touched=True,
    )

    result = villages[["NAME", "SUB_DIST", "CEN_2001"]].copy()
    result["date"] = date_label
    result["mean_mm"] = [s["mean"] for s in stats]
    result["p10_mm"] = [s["percentile_10"] for s in stats]
    result["p50_mm"] = [s["percentile_50"] for s in stats]
    result["p90_mm"] = [s["percentile_90"] for s in stats]
    result["max_mm"] = [s["max"] for s in stats]
    result["pixel_count"] = [s["count"] for s in stats]
    return result


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
