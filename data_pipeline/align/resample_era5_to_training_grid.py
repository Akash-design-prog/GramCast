"""Resample ERA5-Land humidity/wind onto the exact training-pair grid (35x50, south-ascending, matching
fine_rainfall.npy), aggregated to one daily value per variable, aligned to dates.npy by explicit date
lookup (not positional index - ERA5's 2026 file is a known partial year, same gap already documented for
CHIRPS/dates.npy, and date-matching handles that correctly without any special-casing).

Two real bugs this deliberately avoids repeating (see DEV_LOG 2026-09-25 "SEVERE bug found" and the u/v
wind note below):

1. Orientation: ERA5's latitude is DESCENDING (north-first), the opposite of fine_rainfall.npy's
   south-ascending convention - exactly the same mismatch that stacked terrain upside-down against
   rainfall once already. Fixed the same way build_static_terrain() fixes it: reproject onto CHIRPS's
   north-up transform (rasterio's native convention), THEN flip (`arr[:, ::-1, :]`) before trimming to
   the (35, 50) shape, so the final array's row order matches fine_rainfall.npy exactly.

2. Wind is NOT treated as circular here, on purpose. u10/v10 are already Cartesian vector components
   (eastward/northward m/s), not an angle in degrees - unlike aspect (which IS an angle and DOES need
   sin/cos decomposition, see resample_aspect_circular() in resample_to_chirps_grid.py), linearly
   averaging/interpolating u and v components directly is mathematically correct. Converting them to a
   direction-in-degrees first and then needing circular treatment would be introducing a problem that
   using the native components avoids entirely.

A third thing this handles, found while verifying the above: ERA5-Land is a LAND-surface reanalysis and
has genuine, permanent NaN over a handful of cells at the domain's south-west edge (lat 17.7-18.2,
lon 73.0-73.1) in every single year checked - not a resampling bug, real missing source data (almost
certainly open water - a reservoir/lake in that part of the Western Ghats). Confirmed by checking the
RAW ERA5 arrays directly, before any reprojection: 6 cells, identical across 1981/2000/2020/2025.
Left unfilled, this NaN would reproject straight into destination pixels that valid_mask.npy actually
uses for training. Filled via nearest-valid-neighbour before reprojecting (not a global mean - that
would flatten real spatial structure near the coast/reservoir).
"""
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.interpolate import NearestNDInterpolator

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "era5"
TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"
CHIRPS_SAMPLE = Path(__file__).resolve().parents[2] / "data" / "raw" / "chirps" / "chirps_pune_2023_07.nc"
FINE_SHAPE = (35, 50)  # matches fine_rainfall.npy after the trim

# ERA5 variable -> output channel name
VARIABLES = {"d2m": "humidity_proxy", "u10": "wind_u", "v10": "wind_v"}


def get_chirps_grid():
    """Same grid definition as resample_to_chirps_grid.py's get_chirps_grid() - duplicated (not
    imported) since this is the only thing needed from that module and it keeps this script
    self-contained. Returns a north-up transform (row 0 = north edge), matching rasterio's convention."""
    ds = xr.open_dataset(CHIRPS_SAMPLE)
    lat = ds.latitude.values
    lon = ds.longitude.values
    ds.close()

    res_lat = abs(lat[1] - lat[0])
    res_lon = abs(lon[1] - lon[0])
    west = lon[0] - res_lon / 2
    north = lat[-1] + res_lat / 2 if lat[0] < lat[-1] else lat[0] + res_lat / 2
    transform = rasterio.transform.from_origin(west, north, res_lon, res_lat)
    return transform, len(lat), len(lon)


def get_era5_source_transform(lat: np.ndarray, lon: np.ndarray):
    """ERA5's own transform, built from its actual coordinates rather than assumed - confirmed via
    direct inspection that lat[0] is the northernmost value (descending), so lat[0] is the north edge."""
    res_lat = abs(lat[1] - lat[0])
    res_lon = abs(lon[1] - lon[0])
    west = lon[0] - res_lon / 2
    north = lat[0] + res_lat / 2 if lat[0] > lat[-1] else lat[-1] + res_lat / 2
    return rasterio.transform.from_origin(west, north, res_lon, res_lat)


def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    """Fill NaN cells with the nearest valid cell's value (see module docstring - ERA5-Land has real,
    permanent NaN over a few open-water cells in this bbox). A no-op when there's nothing to fill."""
    if not np.isnan(arr).any():
        return arr
    valid = ~np.isnan(arr)
    if not valid.any():
        raise ValueError("entire source array is NaN - cannot fill, something upstream is broken")
    ys, xs = np.indices(arr.shape)
    interp = NearestNDInterpolator(np.column_stack([ys[valid], xs[valid]]), arr[valid])
    filled = arr.copy()
    filled[~valid] = interp(ys[~valid], xs[~valid])
    return filled


def _pad_source_for_bilinear(src_array: np.ndarray, src_transform):
    """Bilinear interpolation cannot extrapolate past the first/last SOURCE CELL CENTRE - a destination
    pixel whose centre falls beyond that (even if still within the source array's outer bbox edge) comes
    back as nodata. Found this for real: 13/1750 destination pixels came back NaN, all a thin strip along
    the south/west domain edge, 7 of which are inside valid_mask.npy (i.e. actually used in the loss) -
    not cosmetic. Fix: edge-replicate the source array by one pixel on every side before reprojecting, so
    bilinear always has a source cell centre beyond every destination pixel we actually need."""
    padded = np.pad(src_array, pad_width=1, mode="edge")
    res_lon, res_lat = src_transform.a, -src_transform.e
    new_west = src_transform.c - res_lon
    new_north = src_transform.f + res_lat
    padded_transform = rasterio.transform.from_origin(new_west, new_north, res_lon, res_lat)
    return padded, padded_transform


def reproject_to_chirps(src_array, src_transform, dst_transform, dst_shape) -> np.ndarray:
    src_array = _fill_nan_nearest(src_array)
    padded_array, padded_transform = _pad_source_for_bilinear(src_array, src_transform)
    dst = np.full(dst_shape, np.nan, dtype="float32")
    reproject(
        source=padded_array.astype("float32"),
        destination=dst,
        src_transform=padded_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,  # upsampling (0.1deg -> 0.05deg) - bilinear, not average (average
        # is for downsampling; and not bicubic, which caused a real negative-overshoot bug on this
        # project already - see DEV_LOG 2026-09-25, bicubic overshoot on coarse_rainfall_bicubic.npy)
    )
    return dst


def build_era5_arrays() -> tuple[dict, np.ndarray]:
    dates = np.load(TRAINING_PAIRS_DIR / "dates.npy")
    T = len(dates)
    date_to_idx = {str(d): i for i, d in enumerate(dates)}

    dst_transform, dst_h, dst_w = get_chirps_grid()
    out = {name: np.full((T, dst_h, dst_w), np.nan, dtype="float32") for name in VARIABLES.values()}
    filled = np.zeros(T, dtype=bool)

    years = sorted(set(int(str(d)[:4]) for d in dates))
    for year in years:
        path = RAW_DIR / f"era5_pune_{year}.nc"
        if not path.exists():
            print(f"WARNING: missing ERA5 file for {year}, leaving those training days as NaN")
            continue

        ds = xr.open_dataset(path)
        lat = ds.latitude.values
        lon = ds.longitude.values
        src_transform = get_era5_source_transform(lat, lon)

        daily = ds.resample(valid_time="1D").mean()
        n_days = daily.valid_time.size
        for t in range(n_days):
            date_str = str(daily.valid_time.values[t])[:10]
            if date_str not in date_to_idx:
                continue  # ERA5 has this day but the training set doesn't - skip, don't assume alignment
            idx = date_to_idx[date_str]
            for var, out_name in VARIABLES.items():
                src_arr = daily[var].values[t]
                out[out_name][idx] = reproject_to_chirps(src_arr, src_transform, dst_transform, (dst_h, dst_w))
            filled[idx] = True
        ds.close()
        print(f"{year}: {n_days} days processed")

    missing = int((~filled).sum())
    if missing:
        print(f"WARNING: {missing}/{T} training-set days have no ERA5 coverage")

    # Orientation fix (see module docstring): flip to south-ascending, then trim to the exact fine shape.
    trimmed = {}
    for name, arr in out.items():
        south_ascending = arr[:, ::-1, :]
        trimmed[name] = south_ascending[:, : FINE_SHAPE[0], : FINE_SHAPE[1]]

    return trimmed, filled


def main():
    arrays, filled = build_era5_arrays()
    out_path = TRAINING_PAIRS_DIR / "era5_humidity_wind.npz"
    np.savez(out_path, valid_mask=filled, **arrays)
    print(f"\nSaved {out_path}")
    for name, arr in arrays.items():
        valid = arr[filled]
        print(
            f"{name}: shape={arr.shape} "
            f"min={np.nanmin(valid):.3f} max={np.nanmax(valid):.3f} "
            f"mean={np.nanmean(valid):.3f} nan_frac_in_filled_days={np.isnan(valid).mean():.4f}"
        )


if __name__ == "__main__":
    main()
