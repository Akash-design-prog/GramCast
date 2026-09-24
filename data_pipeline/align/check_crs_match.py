"""Explicit CRS-match test across every layer in the pipeline.

Per the project's own standing rule (carried over from the GeoEnhance-AI handoff): raster and vector layers
must be confirmed on the same CRS before any zonal-stats overlay is computed, and this must never be assumed -
GeoPandas/rasterio will NOT error loudly if it's wrong, they'll silently misattribute values. CHIRPS in
particular has no explicit CRS attribute in its NetCDF files at all (implicit WGS84 by lat/lon convention only).

Run this after any new layer is added to the pipeline, and again right before building training tensors.
"""
from pathlib import Path

import geopandas as gpd
import rasterio
import xarray as xr

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

EXPECTED_CRS = "EPSG:4326"


def check_vector(path: Path) -> tuple[str, bool]:
    gdf = gpd.read_file(path)
    crs = str(gdf.crs)
    return crs, crs == EXPECTED_CRS


def check_raster(path: Path) -> tuple[str, bool]:
    with rasterio.open(path) as src:
        crs = str(src.crs)
    return crs, crs == EXPECTED_CRS


def check_chirps_netcdf(path: Path) -> tuple[str, bool]:
    ds = xr.open_dataset(path)
    has_lat_lon = "latitude" in ds.coords and "longitude" in ds.coords
    ds.close()
    # CHIRPS ships no explicit CRS attribute - this only confirms the expected coordinate NAMES
    # exist (consistent with lat/lon/WGS84 convention per CHIRPS' own documentation), not a real CRS tag.
    note = "no explicit CRS attribute in file - relying on documented WGS84 convention" if has_lat_lon else "MISSING lat/lon coords"
    return note, has_lat_lon


def main() -> None:
    results = []

    boundary_path = DATA_DIR / "boundaries" / "pune_villages.geojson"
    if boundary_path.exists():
        crs, ok = check_vector(boundary_path)
        results.append(("pune_villages.geojson (vector)", crs, ok))

    for raster_name in ["pune_dem.tif", "pune_slope.tif", "pune_aspect.tif", "pune_worldcover.tif"]:
        raster_path = DATA_DIR / "processed" / raster_name
        if raster_path.exists():
            crs, ok = check_raster(raster_path)
            results.append((raster_name, crs, ok))

    chirps_dir = DATA_DIR / "raw" / "chirps"
    if chirps_dir.exists():
        sample = next(chirps_dir.glob("*.nc"), None)
        if sample is not None:
            note, ok = check_chirps_netcdf(sample)
            results.append((f"chirps ({sample.name}, sample)", note, ok))

    print(f"{'Layer':<40} {'CRS':<45} {'OK'}")
    print("-" * 95)
    all_ok = True
    for name, crs, ok in results:
        print(f"{name:<40} {crs:<45} {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok

    print()
    if all_ok:
        print(f"All {len(results)} layers confirmed on {EXPECTED_CRS} (or documented-equivalent convention).")
    else:
        print("MISMATCH FOUND - fix before running any zonal-stats overlay or training-tensor build.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
