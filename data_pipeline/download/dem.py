"""Download Copernicus GLO-30 DEM tiles covering the Pune pilot bbox and mosaic + clip them.

Source: copernicus-dem-30m.s3.amazonaws.com, public, no auth (confirmed real GeoTIFF via magic-byte check).
One tile per integer degree, named Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.
"""
from pathlib import Path

import rasterio
import requests
from rasterio.mask import mask
from rasterio.merge import merge

BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

# Same padded bbox as chirps.py: 73.0-75.5E, 17.7-19.6N
LON_MIN, LON_MAX = 73.0, 75.5
LAT_MIN, LAT_MAX = 17.7, 19.6

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "dem"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = OUT_DIR / "pune_dem.tif"


def tile_name(lat: int, lon: int) -> str:
    return f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"


def download_tile(lat: int, lon: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    name = tile_name(lat, lon)
    out_path = RAW_DIR / f"{name}.tif"
    if out_path.exists():
        print(f"Skip (exists): {out_path.name}")
        return out_path

    url = f"{BASE_URL}/{name}/{name}.tif"
    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        print(f"  Not found (HTTP {resp.status_code}): {name} - likely ocean/no-data tile, skipping")
        return None
    out_path.write_bytes(resp.content)
    print(f"Downloaded {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main() -> None:
    lats = range(int(LAT_MIN), int(LAT_MAX) + 1)  # 17, 18, 19
    lons = range(int(LON_MIN), int(LON_MAX) + 1)  # 73, 74, 75

    tile_paths = []
    for lat in lats:
        for lon in lons:
            p = download_tile(lat, lon)
            if p is not None:
                tile_paths.append(p)

    if not tile_paths:
        raise RuntimeError("No DEM tiles downloaded")

    print(f"\nMosaicking {len(tile_paths)} tiles...")
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
    )
    for s in srcs:
        s.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_mosaic = OUT_DIR / "_mosaic_tmp.tif"
    with rasterio.open(tmp_mosaic, "w", **meta) as dst:
        dst.write(mosaic)

    print(f"Clipping to bbox ({LON_MIN}, {LAT_MIN}, {LON_MAX}, {LAT_MAX})...")
    bbox_geom = [
        {
            "type": "Polygon",
            "coordinates": [[
                [LON_MIN, LAT_MIN], [LON_MAX, LAT_MIN],
                [LON_MAX, LAT_MAX], [LON_MIN, LAT_MAX], [LON_MIN, LAT_MIN],
            ]],
        }
    ]
    with rasterio.open(tmp_mosaic) as src:
        clipped, clip_transform = mask(src, bbox_geom, crop=True)
        clip_meta = src.meta.copy()
        clip_meta.update(
            height=clipped.shape[1], width=clipped.shape[2], transform=clip_transform
        )

    with rasterio.open(OUT_PATH, "w", **clip_meta) as dst:
        dst.write(clipped)

    tmp_mosaic.unlink()
    print(f"\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"Shape: {clipped.shape}, CRS: {clip_meta['crs']}")
    print(f"Elevation range: {clipped[clipped > -1000].min():.1f} to {clipped.max():.1f} m")


if __name__ == "__main__":
    main()
