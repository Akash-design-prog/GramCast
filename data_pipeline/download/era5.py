"""Download ERA5-Land hourly data for the Pune bbox, Jun-Sep, 1981-2026, via the Copernicus CDS API.

Requires a ~/.cdsapirc file with a valid personal access token (see cds.climate.copernicus.eu/profile) and
the ERA5-Land dataset's licence accepted (cds.climate.copernicus.eu/datasets/reanalysis-era5-land - one-time,
done in-browser, confirmed 2026-09-25).

Real quirk confirmed via a test request: even when 'data_format': 'netcdf' is requested, CDS delivers a ZIP
archive (the actual .nc file inside is typically named data_0.nc) - the file saved by client.retrieve() has a
.nc extension but is actually a zip. Must unzip before reading with xarray.

CDS queues requests server-side (accepted -> running -> successful), unlike the other sources in this pipeline
which stream directly - a single year's request took ~50s end to end in testing, so the full 1981-2026 range
(46 requests) needs to run patiently, not synchronously in one blocking call.

Variables: 2m dewpoint temperature (humidity) and 10m u/v wind components - the guide's priority-3/4 extra
context inputs (section 3 of the project guide). Reduced to synoptic hours (00/06/12/18 UTC) rather than all
24, since the project only needs daily aggregates, not hourly resolution.

*** ORIENTATION WARNING (read before wiring ERA5 into build_training_pairs.py) ***
ERA5's latitude coordinate is DESCENDING (north-first: lat[0]=19.6, lat[-1]=17.7) - the OPPOSITE convention
from CHIRPS/fine_rainfall.npy and IMD, both of which are south-ascending. This is exactly the same orientation
mismatch that was found corrupting terrain_static.npz (see DEV_LOG 2026-09-25, "SEVERE bug found") - that one
stacked terrain data upside-down against rainfall data for the entire training set, undetected until a
ground-truth check caught it. When adding ERA5 as a training channel, flip it first: `data[::-1, :]` on the
latitude axis, so it matches fine_rainfall.npy's row order before anything gets stacked or trimmed together.
"""
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

# Same padded bbox as the rest of the pipeline
LON_MIN, LON_MAX = 73.0, 75.5
LAT_MIN, LAT_MAX = 17.7, 19.6

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "era5"
MONSOON_MONTHS = ["06", "07", "08", "09"]
SYNOPTIC_HOURS = ["00:00", "06:00", "12:00", "18:00"]

VARIABLES = ["2m_dewpoint_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]


def download_year(year: int) -> Path | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"era5_pune_{year}.nc"
    if out_path.exists():
        print(f"Skip (exists): {out_path.name}")
        return out_path

    zip_path = RAW_DIR / f"_era5_pune_{year}_raw.nc"
    client = cdsapi.Client()
    try:
        client.retrieve(
            "reanalysis-era5-land",
            {
                "variable": VARIABLES,
                "year": str(year),
                "month": MONSOON_MONTHS,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": SYNOPTIC_HOURS,
                "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # N, W, S, E
                "data_format": "netcdf",
            },
            str(zip_path),
        )
    except Exception as e:
        print(f"FAILED {year}: {e}")
        return None

    # Unwrap the zip CDS actually delivers, regardless of requested format (see module docstring)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        nc_name = next((n for n in names if n.endswith(".nc")), names[0])
        zf.extract(nc_name, RAW_DIR)
    extracted = RAW_DIR / nc_name
    extracted.rename(out_path)
    zip_path.unlink()

    print(f"Saved {out_path.name} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


def main(start_year: int = 1981, end_year: int = 2026) -> None:
    failed = []
    for year in range(start_year, end_year + 1):
        result = download_year(year)
        if result is None:
            failed.append(year)

    total = end_year - start_year + 1
    print(f"\nDone. {total - len(failed)}/{total} succeeded.")
    if failed:
        print(f"Failed years: {failed}")


if __name__ == "__main__":
    main()
