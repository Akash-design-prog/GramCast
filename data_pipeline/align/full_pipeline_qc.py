"""Consolidated QC: re-runs the key checks already done ad hoc for every source in this pipeline, in one script.

Each check here reproduces a verification that was previously done manually in a DEV_LOG session - see DEV_LOG.md
for the original investigation and reasoning behind each check's specific thresholds. This script exists so
re-verifying the whole pipeline after any change doesn't mean re-typing a dozen ad hoc commands again.
"""
import calendar
import sys
from pathlib import Path

import numpy as np
import xarray as xr

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def check_chirps() -> None:
    print("\n--- CHIRPS ---")
    files = sorted((DATA_DIR / "raw" / "chirps").glob("*.nc"))
    check("CHIRPS file count", len(files) >= 183, f"{len(files)} files")

    bad_shape, bad_days = 0, 0
    for f in files:
        year, month = map(int, f.stem.split("_")[-2:])
        ds = xr.open_dataset(f)
        if ds.sizes.get("latitude") != 38 or ds.sizes.get("longitude") != 50:
            bad_shape += 1
        expected_days = calendar.monthrange(year, month)[1]
        if ds.sizes["time"] != expected_days and not (year == 2026 and month == 9):
            bad_days += 1
        ds.close()
    check("CHIRPS shapes consistent (38x50)", bad_shape == 0, f"{bad_shape} bad")
    check("CHIRPS day counts correct", bad_days == 0, f"{bad_days} mismatched")


def check_imd() -> None:
    print("\n--- IMD ---")
    files = sorted((DATA_DIR / "raw" / "imd_rainfall_pune").glob("*.nc"))
    check("IMD file count", len(files) == 45, f"{len(files)} files (expect 45, 1981-2025)")

    bad_days, has_nan = 0, 0
    for f in files:
        year = int(f.stem.split("_")[-1])
        ds = xr.open_dataset(f)
        expected_days = 366 if calendar.isleap(year) else 365
        if ds.sizes["TIME"] != expected_days:
            bad_days += 1
        if np.isnan(ds["RAINFALL"].values).all():
            has_nan += 1
        ds.close()
    check("IMD day counts correct", bad_days == 0, f"{bad_days} mismatched")
    check("IMD no all-NaN files", has_nan == 0, f"{has_nan} all-NaN")


def check_era5() -> None:
    print("\n--- ERA5 ---")
    files = sorted((DATA_DIR / "raw" / "era5").glob("era5_pune_*.nc"))
    check("ERA5 file count", len(files) == 46, f"{len(files)} files (expect 46, 1981-2026)")

    bad_nan = 0
    for f in files:
        ds = xr.open_dataset(f)
        n_t = ds.sizes["valid_time"]
        for var in ["d2m", "u10", "v10"]:
            nan_count = int(ds[var].isnull().sum())
            if nan_count != 6 * n_t:
                bad_nan += 1
        ds.close()
    check("ERA5 NaN count matches known coastal artifact (6 cells/timestep)", bad_nan == 0, f"{bad_nan} mismatches")


def check_terrain() -> None:
    print("\n--- Terrain (DEM/slope/aspect/WorldCover) ---")
    terrain = np.load(DATA_DIR / "processed" / "training_pairs" / "terrain_static.npz")
    check("terrain_static.npz has all 4 layers", set(terrain.keys()) == {"dem", "slope", "aspect", "worldcover"})
    for k in ["dem", "slope", "aspect"]:
        arr = terrain[k]
        check(f"{k} shape (35,50)", arr.shape == (35, 50), str(arr.shape))
    dem = terrain["dem"]
    check("DEM range sane (0-1600m)", 0 <= dem.min() and dem.max() <= 1600, f"[{dem.min():.0f}, {dem.max():.0f}]")
    slope = terrain["slope"]
    check("slope range sane (0-90deg)", 0 <= slope.min() and slope.max() <= 90, f"[{slope.min():.1f}, {slope.max():.1f}]")
    aspect = terrain["aspect"]
    check("aspect covers full circular range", aspect.min() < 90 and aspect.max() > 270,
          f"[{aspect.min():.1f}, {aspect.max():.1f}] - narrow range would indicate the circular-averaging bug")

    # STRUCTURAL cross-array orientation check - the exact bug class found 2026-09-25.
    # NOTE: an earlier version of this check used a single known point's elevation value with a loose
    # tolerance (abs(diff)<100m) - "test the test" against a deliberately re-broken (upside-down) array
    # found that check would NOT have caught the regression, because the flipped row happened to land on
    # a coincidentally similar elevation (600m vs the correct 563m, both within the 100m tolerance). A
    # value-based proxy check can pass by luck; only a structural check that directly verifies row order
    # is reliable. Fix: recompute the DEM array directly from the aligned raster (same flip+trim logic as
    # build_static_terrain) and compare byte-for-byte against what's actually stored.
    import rasterio
    aligned_path = DATA_DIR / "processed" / "aligned" / "pune_dem.tif"
    with rasterio.open(aligned_path) as src:
        aligned_data = src.read(1)
    expected_dem = aligned_data[::-1, :][:35, :50]  # same flip-then-trim as build_static_terrain
    check(
        "DEM orientation structurally correct (recomputed from aligned raster matches stored array)",
        np.allclose(expected_dem, dem, equal_nan=True),
        f"max diff: {np.nanmax(np.abs(expected_dem - dem)):.2f}m" if not np.allclose(expected_dem, dem, equal_nan=True) else "exact match",
    )


def check_training_pairs() -> None:
    print("\n--- Training pairs ---")
    fine = np.load(DATA_DIR / "processed" / "training_pairs" / "fine_rainfall.npy")
    coarse = np.load(DATA_DIR / "processed" / "training_pairs" / "coarse_rainfall_bicubic.npy")
    dates = np.load(DATA_DIR / "processed" / "training_pairs" / "dates.npy")

    check("fine/coarse shapes match", fine.shape == coarse.shape, str(fine.shape))
    check("no NaN in fine", not np.isnan(fine).any())
    check("no NaN in coarse", not np.isnan(coarse).any())
    check("no negative values in fine", (fine >= 0).all())
    check("no negative values in coarse", (coarse >= 0).all())
    check("5582 samples", len(dates) == 5582, str(len(dates)))

    split_path = DATA_DIR / "processed" / "training_pairs" / "split_indices.npz"
    if split_path.exists():
        splits = np.load(split_path)
        total_assigned = sum(len(splits[k]) for k in ["train", "val", "test"])
        check("split covers all samples", total_assigned == len(dates), f"{total_assigned}/{len(dates)}")
        overlap = set(splits["train"]) & set(splits["val"]) | set(splits["val"]) & set(splits["test"]) | set(splits["train"]) & set(splits["test"])
        check("no split overlap", len(overlap) == 0, f"{len(overlap)} overlapping")
    else:
        check("split_indices.npz exists", False, "not found - run train_val_test_split.py")


def check_crs() -> None:
    print("\n--- CRS match ---")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_crs_match
    try:
        check_crs_match.main()
        check("CRS-match test", True)
    except SystemExit:
        check("CRS-match test", False, "mismatch found - see output above")


def main() -> None:
    check_chirps()
    check_imd()
    check_era5()
    check_terrain()
    check_training_pairs()
    check_crs()

    print(f"\n{'='*50}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks failed: {FAILURES}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
