"""Resample DEM/slope/aspect/WorldCover onto the exact CHIRPS 0.05deg fine grid.

Every layer already shares EPSG:4326 (confirmed by check_crs_match.py), but they're on different native pixel
grids - DEM at ~30m, WorldCover at ~9m, CHIRPS at 0.05deg (~5.5km). Before they can be stacked as channels for
the U-Net they must sit on the SAME grid, cell-for-cell, not just the same CRS.

Continuous fields (DEM, slope, aspect) are resampled with average (area-weighted mean) - appropriate for
downsampling fine to coarse, avoids the aliasing a nearest-neighbour pick would cause on terrain with real
small-scale variation. WorldCover is categorical (land cover class codes), so it uses mode (most common class
in each coarse cell) instead - averaging categorical codes would produce meaningless intermediate numbers.
"""
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import stats

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHIRPS_SAMPLE = Path(__file__).resolve().parents[2] / "data" / "raw" / "chirps" / "chirps_pune_2023_07.nc"


def get_chirps_grid():
    ds = xr.open_dataset(CHIRPS_SAMPLE)
    lat = ds.latitude.values
    lon = ds.longitude.values
    ds.close()

    res_lat = abs(lat[1] - lat[0])
    res_lon = abs(lon[1] - lon[0])
    # CHIRPS coords are cell centres; rasterio transform wants the top-left corner
    west = lon[0] - res_lon / 2
    north = lat[-1] + res_lat / 2 if lat[0] < lat[-1] else lat[0] + res_lat / 2
    lat_ascending = lat[0] < lat[-1]

    transform = rasterio.transform.from_origin(west, north, res_lon, res_lat)
    height, width = len(lat), len(lon)
    return transform, height, width, lat_ascending


def resample_aspect_circular(src_path: Path, out_path: Path) -> None:
    """Aspect is a circular quantity (0deg == 360deg, same direction) - naively area-averaging raw degrees
    is wrong and silently produces the OPPOSITE direction near the wrap boundary (e.g. averaging 355deg and
    5deg, both ~north, gives 180deg = south with a plain mean). Fix: decompose into sin/cos components,
    average those (which behaves correctly across the wrap), then recombine via atan2.
    """
    transform, height, width, lat_ascending = get_chirps_grid()

    with rasterio.open(src_path) as src:
        aspect_rad = np.deg2rad(src.read(1).astype("float64"))
        sin_a = np.sin(aspect_rad).astype("float32")
        cos_a = np.cos(aspect_rad).astype("float32")
        src_transform, src_crs, src_meta = src.transform, src.crs, src.meta.copy()

    sin_dst = np.empty((height, width), dtype="float32")
    cos_dst = np.empty((height, width), dtype="float32")
    for arr, dst in [(sin_a, sin_dst), (cos_a, cos_dst)]:
        reproject(
            source=arr, destination=dst,
            src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
            dst_transform=transform, dst_crs=src_crs, dst_nodata=np.nan,
            resampling=Resampling.average,
        )

    aspect_out = (np.rad2deg(np.arctan2(sin_dst, cos_dst)) + 360) % 360
    if not lat_ascending:
        aspect_out = aspect_out[::-1, :]

    src_meta.update(height=height, width=width, transform=transform, dtype="float32")
    with rasterio.open(out_path, "w", **src_meta) as dst:
        dst.write(aspect_out.astype("float32"), 1)
    print(f"  {out_path.name} (circular-aware): {aspect_out.shape}, range [{aspect_out.min():.2f}, {aspect_out.max():.2f}]")


def resample_layer(src_path: Path, out_path: Path, method: Resampling) -> None:
    transform, height, width, lat_ascending = get_chirps_grid()

    with rasterio.open(src_path) as src:
        dst_array = np.empty((height, width), dtype=src.dtypes[0])
        src_nodata = src.nodata if src.nodata is not None else np.nan
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=transform,
            dst_crs=src.crs,
            dst_nodata=src_nodata,
            resampling=method,
        )
        if not lat_ascending:
            dst_array = dst_array[::-1, :]

        meta = src.meta.copy()
        meta.update(height=height, width=width, transform=transform)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(dst_array, 1)

    print(f"  {out_path.name}: {dst_array.shape}, range [{np.nanmin(dst_array):.2f}, {np.nanmax(dst_array):.2f}]")


def resample_categorical_mode(src_path: Path, out_path: Path) -> None:
    """WorldCover: resample by taking the mode (most common class) per output cell, not averaging."""
    transform, height, width, lat_ascending = get_chirps_grid()

    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    out = np.zeros((height, width), dtype=src_data.dtype)
    inv_transform = ~src_transform
    for row in range(height):
        for col in range(width):
            # Bounds of this output cell in map coords -> source pixel bounds
            x0, y0 = transform * (col, row)
            x1, y1 = transform * (col + 1, row + 1)
            c0, r0 = inv_transform * (x0, y0)
            c1, r1 = inv_transform * (x1, y1)
            r_lo, r_hi = sorted([int(r0), int(r1)])
            c_lo, c_hi = sorted([int(c0), int(c1)])
            r_hi = min(r_hi + 1, src_data.shape[0])
            c_hi = min(c_hi + 1, src_data.shape[1])
            if r_lo >= r_hi or c_lo >= c_hi:
                continue
            window = src_data[r_lo:r_hi, c_lo:c_hi]
            if window.size == 0:
                continue
            mode_result = stats.mode(window, axis=None, keepdims=False)
            out[row, col] = mode_result.mode

    if not lat_ascending:
        out = out[::-1, :]

    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
    meta.update(height=height, width=width, transform=transform)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(out, 1)
    print(f"  {out_path.name}: {out.shape}, unique classes: {sorted(np.unique(out).tolist())}")


def main() -> None:
    OUT_DIR = PROCESSED_DIR / "aligned"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Resampling continuous layers (average) onto CHIRPS grid...")
    for name in ["pune_dem.tif", "pune_slope.tif"]:
        resample_layer(PROCESSED_DIR / name, OUT_DIR / name, Resampling.average)

    print("Resampling aspect (circular-aware) onto CHIRPS grid...")
    resample_aspect_circular(PROCESSED_DIR / "pune_aspect.tif", OUT_DIR / "pune_aspect.tif")

    print("\nResampling WorldCover (mode - categorical) onto CHIRPS grid...")
    resample_categorical_mode(PROCESSED_DIR / "pune_worldcover.tif", OUT_DIR / "pune_worldcover.tif")

    print(f"\nAll layers saved to {OUT_DIR}, aligned to the CHIRPS 0.05deg grid.")


if __name__ == "__main__":
    main()
