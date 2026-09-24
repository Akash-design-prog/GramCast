"""Build coarse/fine training pairs from CHIRPS, per the project guide's method (section 4):
aggregate the fine (0.05deg) CHIRPS grid up to coarse (0.25deg, 5x factor) to make the "block-level" proxy
input, bicubic-upsample it back to fine resolution as the model's actual input, and train to recover the true
fine CHIRPS. Terrain/land-cover channels (already aligned to the CHIRPS grid) are stacked in as static extra
inputs.

Grid note: the padded bbox gives a 38x50 fine grid, and 38 isn't evenly divisible by 5 (0.25/0.05). Trimmed to
35x50 (drops the northernmost 3 rows, ~19.45-19.6N) - checked safe, leaves the real Pune district (17.89-19.39N)
fully covered with margin on both edges.
"""
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from scipy.ndimage import zoom

CHIRPS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "chirps"
ALIGNED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "aligned"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"

COARSEN_FACTOR = 5  # 0.25deg / 0.05deg


def build_static_terrain(fine_shape: tuple[int, int]) -> None:
    """Trim the aligned terrain layers to match the trimmed fine rainfall grid, save once (time-invariant).

    CRITICAL row-order fix (found 2026-09-25 during ground-truth re-verification): the aligned/*.tif rasters
    are in standard rasterio north-up order (row 0 = north), but fine_rainfall.npy comes from CHIRPS' own
    xarray order, which is south-ascending (row 0 = south) - confirmed directly: CHIRPS lat[0]=17.72N (south)
    vs the aligned raster's row 0 = 19.57N (north). Trimming both with the same `[:fine_shape[0]]` slice
    therefore kept OPPOSITE ends of the bbox and left the two arrays in opposite row order entirely - stacking
    them as model input channels without this fix would have spatially mismatched terrain against rainfall by
    ~190km (the full north-south extent of the bbox), with no error and no obviously-wrong-looking output.
    Fix: flip the terrain raster to south-ascending BEFORE trimming, so both arrays end up describing the same
    physical rows in the same order.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = {}
    for name in ["pune_dem.tif", "pune_slope.tif", "pune_aspect.tif", "pune_worldcover.tif"]:
        with rasterio.open(ALIGNED_DIR / name) as src:
            data = src.read(1)
        data_south_ascending = data[::-1, :]  # match fine_rainfall.npy's row order before trimming
        key = name.replace("pune_", "").replace(".tif", "")
        trimmed[key] = data_south_ascending[: fine_shape[0], : fine_shape[1]]
        assert trimmed[key].shape == fine_shape, (
            f"{name}: shape {data.shape} trimmed doesn't match fine grid {fine_shape}"
        )
    np.savez(OUT_DIR / "terrain_static.npz", **trimmed)
    print(f"Saved terrain_static.npz: {list(trimmed.keys())}, each shape {fine_shape}")


def process_all() -> None:
    files = sorted(CHIRPS_DIR.glob("chirps_pune_*.nc"))
    print(f"Processing {len(files)} CHIRPS monthly files...")

    fine_list = []
    coarse_upsampled_list = []
    date_list = []
    static_saved = False

    for f in files:
        ds = xr.open_dataset(f)
        precip = ds["precip"]

        for t in range(precip.sizes["time"]):
            day = precip.isel(time=t)
            fine = day.values  # (38, 50)

            trimmed_shape = (
                (fine.shape[0] // COARSEN_FACTOR) * COARSEN_FACTOR,
                (fine.shape[1] // COARSEN_FACTOR) * COARSEN_FACTOR,
            )
            fine_trimmed = fine[: trimmed_shape[0], : trimmed_shape[1]]

            if not static_saved:
                build_static_terrain(trimmed_shape)
                static_saved = True

            # NaN-safe coarsen: treat NaN (the 6 known coastal-edge cells) as 0 contribution with a mask,
            # matching how CHIRPS' own NaN mask was already characterized as a static ocean-edge artifact.
            # NaN cells are the known static coastal-edge artifact (6 cells, outside actual Pune district -
            # see DEV_LOG 2026-09-24 CHIRPS entry). Fill with 0 for both fine target and coarse aggregation so
            # NaN never silently propagates into a loss computation later, but also save a validity mask so
            # training code can explicitly exclude/downweight these cells rather than being misled into
            # thinking "0 rain" was actually observed there.
            nan_mask = np.isnan(fine_trimmed)
            fine_filled = np.where(nan_mask, 0.0, fine_trimmed)
            valid = (~nan_mask).astype("float32")

            h, w = trimmed_shape
            ch, cw = h // COARSEN_FACTOR, w // COARSEN_FACTOR
            sum_block = fine_filled.reshape(ch, COARSEN_FACTOR, cw, COARSEN_FACTOR).sum(axis=(1, 3))
            count_block = valid.reshape(ch, COARSEN_FACTOR, cw, COARSEN_FACTOR).sum(axis=(1, 3))
            coarse = np.divide(sum_block, count_block, out=np.full_like(sum_block, np.nan), where=count_block > 0)

            coarse_filled = np.nan_to_num(coarse, nan=0.0)
            coarse_upsampled = zoom(coarse_filled, COARSEN_FACTOR, order=3)
            coarse_upsampled = coarse_upsampled[: fine_trimmed.shape[0], : fine_trimmed.shape[1]]
            # Cubic-spline upsampling overshoots/undershoots near sharp gradients (classic ringing artifact) -
            # checked on a test month: 18.3% of pixels came out negative, physically impossible for rainfall.
            # Clip at 0. Note this does NOT fix mass conservation (bicubic isn't expected to conserve it - the
            # guide's own design applies a separate mass-conservation layer to the model's OUTPUT, not this
            # baseline input), it only removes physically-impossible values.
            coarse_upsampled = np.clip(coarse_upsampled, 0.0, None)

            fine_list.append(fine_filled.astype("float32"))
            coarse_upsampled_list.append(coarse_upsampled.astype("float32"))
            date_list.append(str(day.time.values)[:10])

            if not hasattr(process_all, "_valid_mask_saved"):
                np.save(OUT_DIR / "valid_mask.npy", valid.astype("bool"))
                process_all._valid_mask_saved = True

        ds.close()

    fine_arr = np.stack(fine_list)
    coarse_arr = np.stack(coarse_upsampled_list)
    dates_arr = np.array(date_list)

    np.save(OUT_DIR / "fine_rainfall.npy", fine_arr)
    np.save(OUT_DIR / "coarse_rainfall_bicubic.npy", coarse_arr)
    np.save(OUT_DIR / "dates.npy", dates_arr)

    print(f"\nSaved {fine_arr.shape[0]} daily samples.")
    print(f"fine_rainfall.npy: {fine_arr.shape}")
    print(f"coarse_rainfall_bicubic.npy: {coarse_arr.shape}")
    print(f"date range: {dates_arr[0]} to {dates_arr[-1]}")


if __name__ == "__main__":
    process_all()
