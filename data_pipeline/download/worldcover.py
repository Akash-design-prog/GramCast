"""Download ESA WorldCover 10m tiles covering the Pune pilot bbox and mosaic + clip them.

Source: esa-worldcover.s3.eu-central-1.amazonaws.com, public, no auth (confirmed real GeoTIFF via magic-byte
check). Tiles are 3x3 degree blocks, named ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map.tif, where
lat/lon are the tile's SW corner rounded down to a multiple of 3.
"""
from pathlib import Path

import rasterio
import requests
from rasterio.mask import mask
from rasterio.merge import merge

BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

# Same padded bbox as chirps.py / dem.py
LON_MIN, LON_MAX = 73.0, 75.5
LAT_MIN, LAT_MAX = 17.7, 19.6

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "worldcover"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_PATH = OUT_DIR / "pune_worldcover.tif"


def floor_to_3(x: float) -> int:
    return int(x // 3) * 3


def tile_name(lat: int, lon: int) -> str:
    return f"ESA_WorldCover_10m_2021_v200_N{lat:02d}E{lon:03d}_Map"


def download_tile(lat: int, lon: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    name = tile_name(lat, lon)
    out_path = RAW_DIR / f"{name}.tif"
    if out_path.exists():
        print(f"Skip (exists): {out_path.name}")
        return out_path

    url = f"{BASE_URL}/{name}.tif"
    resp = requests.get(url, timeout=180)
    if resp.status_code != 200:
        print(f"  Not found (HTTP {resp.status_code}): {name}, skipping")
        return None
    out_path.write_bytes(resp.content)
    print(f"Downloaded {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def main() -> None:
    sw_lats = sorted({floor_to_3(LAT_MIN), floor_to_3(LAT_MAX)})
    sw_lons = sorted({floor_to_3(LON_MIN), floor_to_3(LON_MAX)})

    tile_paths = []
    for lat in sw_lats:
        for lon in sw_lons:
            p = download_tile(lat, lon)
            if p is not None:
                tile_paths.append(p)

    if not tile_paths:
        raise RuntimeError("No WorldCover tiles downloaded")

    # Clip each tile to the bbox BEFORE merging - each full tile is 36000x36000px (~9m pixels),
    # far larger than our bbox needs; mosaicking at full resolution first exhausts memory.
    bbox_geom = [{
        "type": "Polygon",
        "coordinates": [[
            [LON_MIN, LAT_MIN], [LON_MAX, LAT_MIN],
            [LON_MAX, LAT_MAX], [LON_MIN, LAT_MAX], [LON_MIN, LAT_MIN],
        ]],
    }]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clipped_paths = []
    for p in tile_paths:
        with rasterio.open(p) as src:
            try:
                clipped, clip_transform = mask(src, bbox_geom, crop=True)
            except ValueError:
                print(f"  {p.name}: no overlap with bbox, skipping")
                continue
            clip_meta = src.meta.copy()
            clip_meta.update(
                height=clipped.shape[1], width=clipped.shape[2], transform=clip_transform
            )
        clipped_path = OUT_DIR / f"_wc_clip_{p.stem}.tif"
        with rasterio.open(clipped_path, "w", **clip_meta) as dst:
            dst.write(clipped)
        clipped_paths.append(clipped_path)
        print(f"  Clipped {p.name}: {clipped.shape}")

    print(f"\nMerging {len(clipped_paths)} clipped tiles...")
    srcs = [rasterio.open(p) for p in clipped_paths]
    mosaic, transform = merge(srcs)
    clip_meta = srcs[0].meta.copy()
    clip_meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    for s in srcs:
        s.close()

    with rasterio.open(OUT_PATH, "w", **clip_meta) as dst:
        dst.write(mosaic)
    clipped = mosaic

    for p in clipped_paths:
        p.unlink()
    print(f"\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"Shape: {clipped.shape}, CRS: {clip_meta['crs']}")

    import numpy as np
    vals, counts = np.unique(clipped, return_counts=True)
    print("Land cover class pixel counts:", dict(zip(vals.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
