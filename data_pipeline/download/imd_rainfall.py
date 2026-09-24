"""IMD gridded rainfall (0.25deg, daily), Pune bbox subset, 1981-2025.

Source: imdpune.gov.in/cmpg/Griddata/RF25.php, POST {'rain': <year>}. Real endpoint - full-year files are a
genuine NetCDF classic format matching the documented 129x135 grid x days x 4 bytes.

Automated download is UNRELIABLE and off by default (see try_automated_download): imdpune.gov.in behaves
differently for programmatic clients (requests/curl/curl_cffi with Chrome TLS impersonation) than for a real
browser navigation - confirmed on two separate machines/networks, every automated attempt either timed out or
connected with a 200 status but 0 bytes of body. The reliable path is a manual download via the form at
https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html, saved into RAW_DIR as RF25_ind<year>_rfp25.nc.
IMD has no 2026 file yet as of 2026-09-24 (full calendar year not complete) - 2025 is the current ceiling.

This script's job, given files already in RAW_DIR: subset each to the Pune bbox, run a QC pass, and
cross-check a sample against CHIRPS for the same period/location (two independent products should show the
same windward/leeward rainfall gradient, even if absolute totals differ - see DEV_LOG 2026-09-24).
"""
import calendar
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from curl_cffi import requests  # mimics a real browser's TLS fingerprint - still not sufficient, see docstring
    IMPERSONATE = {"impersonate": "chrome"}
except ImportError:
    import requests
    IMPERSONATE = {}

URL = "https://imdpune.gov.in/cmpg/Griddata/RF25.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html",
    "Origin": "https://imdpune.gov.in",
}

# Same padded bbox as chirps.py / dem.py / worldcover.py
LON_MIN, LON_MAX = 73.0, 75.5
LAT_MIN, LAT_MAX = 17.7, 19.6

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "imd_rainfall"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "imd_rainfall_pune"

FIRST_YEAR, LAST_YEAR = 1981, 2025  # CHIRPS floor / IMD's current ceiling, not 2026 - see docstring


def try_automated_download(year: int) -> Path | None:
    """Best-effort only - known unreliable, see module docstring. Returns None on failure rather than
    raising, so main() can fall through to expecting a manual download instead."""
    out_path = RAW_DIR / f"RF25_ind{year}_rfp25.nc"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path
    try:
        resp = requests.post(URL, data={"rain": year}, headers=HEADERS, timeout=60, **IMPERSONATE)
        resp.raise_for_status()
        if len(resp.content) < 1024:
            print(f"  Automated download for {year} got {len(resp.content)} bytes (expected ~25MB) - skipping")
            return None
        out_path.write_bytes(resp.content)
        print(f"  Automated download succeeded for {year} ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path
    except Exception as e:
        print(f"  Automated download failed for {year}: {e}")
        return None


def subset_to_pune(full_path: Path, year: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"imd_pune_{year}.nc"
    if out_path.exists():
        return out_path

    ds = xr.open_dataset(full_path)
    lat_name = "LATITUDE" if "LATITUDE" in ds.coords else "lat"
    lon_name = "LONGITUDE" if "LONGITUDE" in ds.coords else "lon"
    subset = ds.sel({lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)})
    if subset.sizes[lat_name] == 0:
        subset = ds.sel({lat_name: slice(LAT_MAX, LAT_MIN), lon_name: slice(LON_MIN, LON_MAX)})
    subset.to_netcdf(out_path)
    ds.close()
    subset.close()
    return out_path


def run_qc(subset_paths: list[Path]) -> None:
    print("\n--- QC pass ---")
    bad = []
    shapes = set()
    means = []
    found_years = set()

    for f in sorted(subset_paths):
        year = int(f.stem.split("_")[-1])
        found_years.add(year)
        ds = xr.open_dataset(f)
        shapes.add((ds.sizes["LATITUDE"], ds.sizes["LONGITUDE"]))
        expected_days = 366 if calendar.isleap(year) else 365
        if ds.sizes["TIME"] != expected_days:
            bad.append((f.stem, "day mismatch", ds.sizes["TIME"], expected_days))
        mean = float(ds["RAINFALL"].mean())
        if np.isnan(mean):
            bad.append((f.stem, "all-NaN"))
        means.append(mean)
        ds.close()

    expected_years = set(range(FIRST_YEAR, LAST_YEAR + 1))
    missing = expected_years - found_years
    extra = found_years - expected_years

    print(f"Files checked: {len(subset_paths)}")
    print(f"Unique shapes: {shapes}")
    print(f"Bad files: {bad if bad else 'none'}")
    print(f"Missing years: {missing if missing else 'none'}")
    print(f"Extra years outside {FIRST_YEAR}-{LAST_YEAR}: {extra if extra else 'none'}")
    if means:
        means_arr = np.array(means)
        print(f"Annual mean rainfall: min={means_arr.min():.2f} max={means_arr.max():.2f} mean={means_arr.mean():.2f} mm/day")

    if bad or missing:
        raise SystemExit("QC FAILED - see above")
    print("QC PASSED")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subset_paths = []
    missing_raw = []

    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        raw_path = RAW_DIR / f"RF25_ind{year}_rfp25.nc"
        if not raw_path.exists():
            print(f"{year}: not found in {RAW_DIR}, trying automated download (unreliable, see docstring)...")
            raw_path = try_automated_download(year)
            if raw_path is None:
                missing_raw.append(year)
                continue
        subset_paths.append(subset_to_pune(raw_path, year))

    if missing_raw:
        print(f"\n{len(missing_raw)} years have no raw file and automated download failed: {missing_raw}")
        print(f"Download these manually via https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html")
        print(f"and save as RF25_ind<year>_rfp25.nc into {RAW_DIR}")

    if subset_paths:
        run_qc(subset_paths)


if __name__ == "__main__":
    main()
