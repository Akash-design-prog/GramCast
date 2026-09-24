"""Derive slope and aspect rasters from the Pune DEM.

Input: data/processed/pune_dem.tif (EPSG:4326, from data_pipeline/download/dem.py).
Slope/aspect computed via Horn's method on a locally-projected (UTM 43N) copy, since slope in degrees needs
real distance units (metres), not degrees of lat/lon - a common silent-failure trap noted in the handoff doc.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

IN_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "pune_dem.tif"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
UTM_CRS = "EPSG:32643"  # UTM 43N, covers Pune district


def reproject_to_utm(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, UTM_CRS, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(crs=UTM_CRS, transform=transform, width=width, height=height)

        with rasterio.open(dst_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=UTM_CRS,
                resampling=Resampling.bilinear,
            )


def compute_slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    # Horn's method (3x3 kernel), res = pixel size in metres
    dzdx = (
        (dem[:-2, 2:] + 2 * dem[1:-1, 2:] + dem[2:, 2:])
        - (dem[:-2, :-2] + 2 * dem[1:-1, :-2] + dem[2:, :-2])
    ) / (8 * res)
    dzdy = (
        (dem[2:, :-2] + 2 * dem[2:, 1:-1] + dem[2:, 2:])
        - (dem[:-2, :-2] + 2 * dem[:-2, 1:-1] + dem[:-2, 2:])
    ) / (8 * res)

    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = np.where(aspect < 0, 90.0 - aspect, np.where(aspect > 90, 360.0 - aspect + 90, 90.0 - aspect))

    pad = np.full((1, slope.shape[1]), np.nan)
    slope = np.vstack([pad, slope, pad])
    aspect = np.vstack([pad, aspect, pad])
    pad_col = np.full((slope.shape[0], 1), np.nan)
    slope = np.hstack([pad_col, slope, pad_col])
    aspect = np.hstack([pad_col, aspect, pad_col])
    return slope, aspect


def main() -> None:
    utm_dem_path = OUT_DIR / "_pune_dem_utm.tif"
    print("Reprojecting DEM to UTM 43N for metric slope calc...")
    reproject_to_utm(IN_PATH, utm_dem_path)

    with rasterio.open(utm_dem_path) as src:
        dem = src.read(1)
        res = src.res[0]
        meta = src.meta.copy()
        print(f"UTM DEM shape: {dem.shape}, resolution: {res:.1f}m")

    slope, aspect = compute_slope_aspect(dem, res)

    meta.update(dtype="float32", count=1, nodata=np.nan)
    utm_slope_path = OUT_DIR / "_pune_slope_utm.tif"
    utm_aspect_path = OUT_DIR / "_pune_aspect_utm.tif"
    with rasterio.open(utm_slope_path, "w", **meta) as dst:
        dst.write(slope.astype("float32"), 1)
    with rasterio.open(utm_aspect_path, "w", **meta) as dst:
        dst.write(aspect.astype("float32"), 1)

    utm_dem_path.unlink()

    # Reproject back to EPSG:4326 to match every other raster/vector layer in the pipeline -
    # slope/aspect must stay in the same CRS as the DEM/WorldCover/boundaries, not be left in UTM.
    print("Reprojecting slope/aspect back to EPSG:4326...")
    with rasterio.open(IN_PATH) as ref:
        target_crs = ref.crs
        target_transform = ref.transform
        target_width, target_height = ref.width, ref.height

    for utm_path, out_name in [(utm_slope_path, "pune_slope.tif"), (utm_aspect_path, "pune_aspect.tif")]:
        with rasterio.open(utm_path) as src:
            out_meta = src.meta.copy()
            out_meta.update(crs=target_crs, transform=target_transform, width=target_width, height=target_height)
            with rasterio.open(OUT_DIR / out_name, "w", **out_meta) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=np.nan,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
        utm_path.unlink()

    valid_slope = slope[~np.isnan(slope)]
    print(f"\nSlope range: {valid_slope.min():.2f} to {valid_slope.max():.2f} deg, mean {valid_slope.mean():.2f}")
    print(f"Saved: {OUT_DIR / 'pune_slope.tif'}, {OUT_DIR / 'pune_aspect.tif'}")


if __name__ == "__main__":
    main()
