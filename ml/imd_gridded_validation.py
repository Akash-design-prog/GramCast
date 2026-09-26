"""Cross-validation against IMD's independently-gridded rainfall product - the closest available
substitute for the guide's "independent station validation" requirement.

IMPORTANT HONESTY NOTE: this is NOT true station validation. The guide calls for real point-level
IMD rain-gauge station observations, never used in training. We don't have that data source - IMD's
station-observation portal blocks programmatic access (same blocker already documented for the
gridded product's download page), and no manual acquisition has been done. What IS available and
genuinely independent: IMD's own 0.25deg GRIDDED rainfall product (`data/raw/imd_rainfall_pune/`),
built by IMD from station observations via a DIFFERENT interpolation methodology than CHIRPS (which
is a satellite+station blend). Comparing our model's predictions against IMD-gridded is a legitimate
second-source cross-check - agreement between two independently-constructed products is meaningful
signal - but it is explicitly weaker evidence than true point-station validation, and is reported as
such, not conflated with it.

Grid alignment: IMD is 0.25deg (8x11 over this bbox), our fine grid is 0.05deg (35x50) - fine
predictions are area-averaged (Resampling.average, matching the convention already used for
DEM/WorldCover downsampling elsewhere in this pipeline) onto IMD's exact grid before comparing.
Both grids use the same south-ascending row convention (IMD's own LATITUDE coordinate is ascending,
conveniently matching fine_rainfall.npy's convention already - no flip needed here, unlike ERA5).

Date range: IMD gridded data covers 1981-2025 (45 years) vs our test split's 2021-2026 - only the
2021-2025 overlap can be checked (2026 has no IMD file yet).

Usage: python ml/imd_gridded_validation.py --checkpoint ml/checkpoints/<run>_best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline" / "align"))
from dataset import load_datasets
from unet import ResidualUNet
from preprocessing import ERA5_CHANNELS
from metrics import rmse, mae, correlation, _check_no_nan
from resample_era5_to_training_grid import get_chirps_grid

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "training_pairs"
IMD_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "imd_rainfall_pune"
FINE_SHAPE = (35, 50)


def get_fine_grid_south_ascending_transform():
    """Same grid as get_chirps_grid() (north-up, 38 rows), re-expressed as a south-ascending transform
    (row 0 = south edge, row increases northward) for the TRIMMED 35-row fine grid - trimming removed
    the northernmost 3 rows, so the south edge is unchanged from the full 38-row grid."""
    transform_northup, h_full, w = get_chirps_grid()
    west = transform_northup.c
    north_full = transform_northup.f
    res = transform_northup.a
    south = north_full - h_full * res
    return rasterio.transform.from_origin(west, south, res, -res)


def get_imd_south_ascending_transform(lat, lon):
    res = abs(lat[1] - lat[0])
    west = lon[0] - res / 2
    south = lat[0] - res / 2
    return rasterio.transform.from_origin(west, south, res, -res), len(lat), len(lon)


def aggregate_to_imd_grid(fine_array: np.ndarray, src_transform, dst_transform, dst_shape) -> np.ndarray:
    """fine_array: (35, 50), south-ascending. Returns (dst_h, dst_w), south-ascending, matching IMD's own array order."""
    dst = np.full(dst_shape, np.nan, dtype="float32")
    reproject(
        source=fine_array.astype("float32"),
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.average,  # downsampling fine->coarse - area-weighted average, matching DEM/WorldCover convention
    )
    return dst


def get_unet_predictions_for_dates(model, ds, channel_indices, device) -> np.ndarray:
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=64, shuffle=False)
    p50_all = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"]
            if channel_indices is not None:
                x = x[:, channel_indices]
            x = x.to(device)
            bicubic_raw = batch["bicubic_raw"].to(device)
            coarse_nn_raw = batch["coarse_nn_raw"].to(device)
            q = model.predict_quantiles(x, bicubic_raw, coarse_nn_raw)
            p50_all.append(q["p50"].cpu().numpy())
    return np.concatenate(p50_all)[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("args") or ckpt.get("config")

    dates = np.load(TRAINING_PAIRS_DIR / "dates.npy")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    test_idx = splits["test"]
    test_dates = dates[test_idx]

    imd_files = sorted(IMD_DIR.glob("imd_pune_*.nc"))
    imd_years = {int(p.stem.split("_")[-1]) for p in imd_files}
    overlap_mask = np.array([int(str(d)[:4]) in imd_years for d in test_dates])
    n_overlap = int(overlap_mask.sum())
    print(f"Test set: {len(test_idx)} days. IMD-gridded overlap (years with an IMD file): {n_overlap} days.")
    if n_overlap == 0:
        raise RuntimeError("no overlap between the test split and available IMD-gridded years - nothing to validate")

    train_ds, val_ds, test_ds, stats, valid_mask = load_datasets()
    full_channels = train_ds.input_tensor.shape[1]
    ckpt_in_channels = ckpt["model_state"]["enc1.block.0.weight"].shape[1]
    channel_indices = None
    if ckpt_in_channels != full_channels:
        if ckpt_in_channels == full_channels - len(ERA5_CHANNELS):
            channel_indices = [0] + list(range(1 + len(ERA5_CHANNELS), full_channels))
        else:
            raise ValueError(f"checkpoint expects {ckpt_in_channels} channels, unrecognized (current={full_channels})")

    model = ResidualUNet(in_channels=ckpt_in_channels, base_channels=ckpt_config["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print("Running U-Net over the full test set...")
    p50 = get_unet_predictions_for_dates(model, test_ds, channel_indices, device)

    coarse_bicubic = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
    bicubic_test = coarse_bicubic[test_idx]

    src_transform = get_fine_grid_south_ascending_transform()

    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    fine_test = fine[test_idx]

    model_agg, bicubic_agg, chirps_truth_agg, imd_vals = [], [], [], []
    years_processed = []
    for year in sorted(imd_years):
        year_mask = np.array([str(d).startswith(str(year)) for d in test_dates]) & overlap_mask
        if not year_mask.any():
            continue
        ds_imd = xr.open_dataset(IMD_DIR / f"imd_pune_{year}.nc")
        imd_lat, imd_lon = ds_imd.LATITUDE.values, ds_imd.LONGITUDE.values
        dst_transform, dst_h, dst_w = get_imd_south_ascending_transform(imd_lat, imd_lon)

        year_dates = test_dates[year_mask]
        year_p50 = p50[year_mask]
        year_bicubic = bicubic_test[year_mask]
        year_fine_truth = fine_test[year_mask]
        for i, date_str in enumerate(year_dates):
            imd_day = ds_imd["RAINFALL"].sel(TIME=str(date_str)).values
            if np.isnan(imd_day).any():
                continue
            model_agg.append(aggregate_to_imd_grid(year_p50[i], src_transform, dst_transform, (dst_h, dst_w)))
            bicubic_agg.append(aggregate_to_imd_grid(year_bicubic[i], src_transform, dst_transform, (dst_h, dst_w)))
            chirps_truth_agg.append(aggregate_to_imd_grid(year_fine_truth[i], src_transform, dst_transform, (dst_h, dst_w)))
            imd_vals.append(imd_day)
        ds_imd.close()
        years_processed.append(year)

    model_agg = np.stack(model_agg)
    bicubic_agg = np.stack(bicubic_agg)
    chirps_truth_agg = np.stack(chirps_truth_agg)
    imd_vals = np.stack(imd_vals).astype("float32")
    print(f"Compared {model_agg.shape[0]} days across years {years_processed}, grid shape {model_agg.shape[1:]}")

    mask = np.ones(model_agg.shape[1:], dtype=bool)  # IMD's own grid has no nodata cells over this bbox (confirmed 0% NaN earlier)
    print(f"\n{'Metric':<20}{'model_p50_vs_IMD':>20}{'bicubic_vs_IMD':>20}{'CHIRPS_truth_vs_IMD':>22}")
    print(f"{'RMSE':<20}{rmse(model_agg, imd_vals, mask):>20.4f}{rmse(bicubic_agg, imd_vals, mask):>20.4f}{rmse(chirps_truth_agg, imd_vals, mask):>22.4f}")
    print(f"{'MAE':<20}{mae(model_agg, imd_vals, mask):>20.4f}{mae(bicubic_agg, imd_vals, mask):>20.4f}{mae(chirps_truth_agg, imd_vals, mask):>22.4f}")
    print(f"{'correlation':<20}{correlation(model_agg, imd_vals, mask):>20.4f}{correlation(bicubic_agg, imd_vals, mask):>20.4f}{correlation(chirps_truth_agg, imd_vals, mask):>22.4f}")
    print("\nCHIRPS_truth_vs_IMD is the CEILING for this comparison - it's what the actual training target (CHIRPS)")
    print("itself agrees with IMD at, with zero model involved. The model can't be expected to beat its own training")
    print("target's agreement with an independent product it was never trained to match. Read model_p50_vs_IMD")
    print("relative to that ceiling, not in isolation - a number close to the ceiling means the model is doing about")
    print("as well as is possible given how much CHIRPS and IMD already disagree with each other.")
    print("\nReminder: this is a cross-source consistency check against IMD's independently-gridded product,")
    print("NOT true point-station validation (no raw station data available) - report it as that, explicitly.")


if __name__ == "__main__":
    main()
