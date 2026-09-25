"""Stress test: run zonal_stats-equivalent computation across every single day in the training set.

Performance fix: the original version called rasterstats.zonal_stats() once per day, which re-rasterizes all
2,003 village polygons against the grid on every single call - the polygons never change, only the day's
values do. That took ~8.8s/day (projected ~13.7 hours for all 5,582 days). Fixed: rasterize each village's
pixel membership ONCE (via rasterstats itself, with raster_out=True, on a dummy grid) and reuse those
precomputed pixel index lists for the per-day aggregation with plain numpy - the daily loop becomes pure array
indexing, not repeated polygon rasterization.
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zonal_stats import get_chirps_grid_transform, BOUNDARIES_PATH

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"


def precompute_village_pixel_masks(villages, transform, grid_shape):
    """Rasterize each village's touched-pixel mask once, using a dummy all-ones grid + raster_out=True to
    extract which pixel (row,col) indices each polygon actually touches under all_touched=True."""
    dummy = np.ones(grid_shape, dtype="float32")
    stats = zonal_stats(
        villages, dummy, affine=transform, stats=["count"], all_touched=True, raster_out=True, nodata=-9999,
    )
    masks = []
    for s in stats:
        mini_raster = s["mini_raster_array"]  # masked array over the polygon's bounding window
        window = s["mini_raster_affine"]
        # Figure out which (row, col) in the FULL grid this mini raster's unmasked cells correspond to
        if mini_raster is None or mini_raster.mask.all():
            masks.append((np.array([], dtype=int), np.array([], dtype=int)))
            continue
        rows_local, cols_local = np.where(~mini_raster.mask)
        # convert local mini-raster pixel to full-grid row/col via the transform offset
        full_col0 = round((window.c - transform.c) / transform.a)
        full_row0 = round((window.f - transform.f) / transform.e)
        full_rows = rows_local + full_row0
        full_cols = cols_local + full_col0
        valid = (full_rows >= 0) & (full_rows < grid_shape[0]) & (full_cols >= 0) & (full_cols < grid_shape[1])
        masks.append((full_rows[valid], full_cols[valid]))
    return masks


def main():
    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    dates = np.load(TRAINING_PAIRS_DIR / "dates.npy")
    villages = gpd.read_file(BOUNDARIES_PATH)
    transform, crs = get_chirps_grid_transform()
    grid_shape = fine.shape[1:]

    print("Precomputing village pixel masks (once)...", flush=True)
    t_pre = time.time()
    masks = precompute_village_pixel_masks(villages, transform, grid_shape)
    print(f"Precompute done in {time.time()-t_pre:.1f}s", flush=True)

    n_zero_pixel = sum(1 for r, c in masks if len(r) == 0)
    print(f"Villages with zero pixels after precompute: {n_zero_pixel} (sanity check vs earlier all_touched=True result: expect 0)")

    t0 = time.time()
    crashes = []
    coverage_failures = []
    negative_found = []
    nan_found = []
    extreme_found = []

    n = len(dates)
    for i in range(n):
        day_grid = fine[i][::-1, :]  # north-up, matches the mask row/col convention
        try:
            means = np.full(len(masks), np.nan)
            for vi, (rows, cols) in enumerate(masks):
                if len(rows) == 0:
                    continue
                means[vi] = day_grid[rows, cols].mean()
        except Exception as e:
            crashes.append((dates[i], str(e)))
            continue

        zero_cov = sum(1 for r, c in masks if len(r) == 0)
        if zero_cov:
            coverage_failures.append((dates[i], zero_cov))
        if np.any(means[~np.isnan(means)] < 0):
            negative_found.append(dates[i])
        if np.isnan(means).any() and zero_cov == 0:
            nan_found.append(dates[i])
        if np.any(np.nan_to_num(means) > 600):
            extreme_found.append((dates[i], float(np.nanmax(means))))

        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(
                f"[{i+1}/{n}] {elapsed:.1f}s elapsed, {rate:.1f} days/s, "
                f"ETA {eta:.1f}s | crashes={len(crashes)} cov_fail={len(coverage_failures)} "
                f"neg={len(negative_found)} nan={len(nan_found)} extreme={len(extreme_found)}",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\n=== DONE: {n} days in {elapsed:.1f}s ({n/elapsed:.1f} days/s) ===")
    print(f"Crashes: {len(crashes)}", crashes[:5] if crashes else "")
    print(f"Coverage failures: {len(coverage_failures)}", coverage_failures[:5] if coverage_failures else "")
    print(f"Negative values: {len(negative_found)}", negative_found[:5] if negative_found else "")
    print(f"NaN values (excluding known zero-pixel villages): {len(nan_found)}", nan_found[:5] if nan_found else "")
    print(f"Extreme values (>600mm village mean): {len(extreme_found)}", extreme_found[:10] if extreme_found else "")


if __name__ == "__main__":
    main()
