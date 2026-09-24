"""Download CHIRPS 0.05deg daily rainfall for the Pune pilot bbox, Jun-Sep, 1981-2026.

Source: data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/by_month/ (confirmed real, no login).
Each monthly file is global (~86MB); we download it, subset to the Pune bbox, save the small subset, and
discard the global file rather than keeping ~8GB of data we don't need.

Throughput per connection was measured at ~286 KB/s (single-file curl test, 17.1MB/60s); 3 concurrent
connections gave ~1.8MB/s combined, so the bottleneck is per-connection, not the pipe - download in parallel.
"""
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import xarray as xr

MAX_WORKERS = 6

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/by_month"

# Pune district bounds are 73.32-75.16E, 17.89-19.39N (extract_pune_district.py) - padded for aggregation headroom
LON_MIN, LON_MAX = 73.0, 75.5
LAT_MIN, LAT_MAX = 17.7, 19.6

MONSOON_MONTHS = (6, 7, 8, 9)

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "chirps"


def download_and_subset(year: int, month: int, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chirps_pune_{year}_{month:02d}.nc"
    if out_path.exists():
        print(f"Skip (exists): {out_path.name}")
        return out_path

    url = f"{BASE_URL}/chirps-v2.0.{year}.{month:02d}.days_p05.nc"

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(resp.raw, f)

        ds = xr.open_dataset(tmp_path)
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"

        subset = ds.sel(
            {lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)}
        )
        if subset.sizes[lat_name] == 0 or subset.sizes[lon_name] == 0:
            subset = ds.sel(
                {lat_name: slice(LAT_MAX, LAT_MIN), lon_name: slice(LON_MIN, LON_MAX)}
            )
        if subset.sizes[lat_name] == 0 or subset.sizes[lon_name] == 0:
            raise RuntimeError(f"Empty subset for {year}-{month:02d} - check lat ordering/bbox")

        subset.to_netcdf(out_path)
        ds.close()
        subset.close()
        print(f"Saved {out_path.name}: {subset.sizes} ({out_path.stat().st_size / 1e6:.2f} MB)")
    finally:
        tmp_path.unlink(missing_ok=True)

    return out_path


def main(start_year: int = 1981, end_year: int = 2026, max_workers: int = MAX_WORKERS) -> None:
    jobs = [
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in MONSOON_MONTHS
    ]
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_and_subset, y, m): (y, m) for y, m in jobs}
        for fut in as_completed(futures):
            year, month = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"FAILED {year}-{month:02d}: {e}")
                failed.append((year, month))

    print(f"\nDone. {len(jobs) - len(failed)}/{len(jobs)} succeeded.")
    if failed:
        print(f"Failed: {sorted(failed)}")


if __name__ == "__main__":
    main()
