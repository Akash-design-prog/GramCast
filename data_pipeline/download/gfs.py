"""Fetch the latest GFS 0.25deg forecast (precipitation) for the Pune bbox via idx-based byte-range subsetting.

Source: noaa-gfs-bdp-pds.s3.amazonaws.com, public, no auth (confirmed real GRIB2 via magic-byte check on a
range-requested chunk). Avoids downloading full ~500MB pgrb2 files by using each file's .idx sidecar to find
the byte offset of just the variable we want (APCP = accumulated precipitation).
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "gfs"

VARIABLE = "APCP"


def latest_available_run(max_lookback_hours: int = 12) -> tuple[str, str]:
    """GFS runs at 00/06/12/18z; find the most recent one that actually has files posted."""
    now = datetime.now(timezone.utc)
    for hours_back in range(0, max_lookback_hours + 1, 1):
        candidate = now - timedelta(hours=hours_back)
        cycle_hour = (candidate.hour // 6) * 6
        run_date = candidate.strftime("%Y%m%d")
        run_hour = f"{cycle_hour:02d}"
        idx_url = f"{BASE_URL}/gfs.{run_date}/{run_hour}/atmos/gfs.t{run_hour}z.pgrb2.0p25.f000.idx"
        resp = requests.head(idx_url, timeout=15)
        if resp.status_code == 200:
            return run_date, run_hour
    raise RuntimeError("No recent GFS run found - check network or NOMADS/S3 status")


def fetch_variable_bytes(
    run_date: str, run_hour: str, forecast_hour: int, variable: str, accumulation: str = "cumulative"
) -> bytes:
    """accumulation: 'cumulative' picks the '0-N day acc fcst' entry (total rain since forecast start -
    this is what you want for computing daily totals by differencing consecutive forecast hours). 'incremental'
    picks the shorter windowed entry (e.g. '42-48 hour acc fcst') instead - rarely what you want for daily data.
    GFS APCP idx lines at 24h-multiple forecast hours have BOTH; picking the wrong one silently gives 6-hour
    rain instead of a day's rain, so this is explicit rather than "whichever line matches first".
    """
    fhr = f"f{forecast_hour:03d}"
    grib_url = f"{BASE_URL}/gfs.{run_date}/{run_hour}/atmos/gfs.t{run_hour}z.pgrb2.0p25.{fhr}"
    idx_url = f"{grib_url}.idx"

    idx_resp = requests.get(idx_url, timeout=30)
    idx_resp.raise_for_status()
    lines = idx_resp.text.strip().split("\n")

    apcp_lines = [(i, line) for i, line in enumerate(lines) if f":{variable}:" in line]
    if not apcp_lines:
        raise ValueError(f"{variable} not found in idx for {fhr}")

    match_line_num = None
    if accumulation == "cumulative":
        for i, line in apcp_lines:
            if "day acc fcst" in line and line.split(":")[-2].startswith("0-"):
                match_line_num = i
                break
    if match_line_num is None:
        # fall back to first match (incremental window) if no cumulative entry exists at this fhr
        match_line_num = apcp_lines[0][0]
    print(f"    idx line used: {lines[match_line_num]}")

    start_byte = int(lines[match_line_num].split(":")[1])
    if match_line_num + 1 < len(lines):
        end_byte = int(lines[match_line_num + 1].split(":")[1]) - 1
        range_header = f"bytes={start_byte}-{end_byte}"
    else:
        range_header = f"bytes={start_byte}-"

    resp = requests.get(grib_url, headers={"Range": range_header}, timeout=90)
    resp.raise_for_status()
    return resp.content


def main(forecast_hours: tuple[int, ...] = tuple(range(24, 121, 24))) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_date, run_hour = latest_available_run()
    print(f"Using GFS run: {run_date} {run_hour}z")

    for fhr in forecast_hours:
        out_path = OUT_DIR / f"gfs_{run_date}_{run_hour}z_f{fhr:03d}_{VARIABLE}.grib2"
        if out_path.exists():
            print(f"Skip (exists): {out_path.name}")
            continue
        try:
            data = fetch_variable_bytes(run_date, run_hour, fhr, VARIABLE)
            out_path.write_bytes(data)
            print(f"Saved {out_path.name} ({len(data) / 1e3:.1f} KB), magic: {data[:4]}")
        except Exception as e:
            print(f"FAILED f{fhr:03d}: {e}")


if __name__ == "__main__":
    main()
