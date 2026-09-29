"""Fetches the runtime data bundle (training-pair arrays, best checkpoint, CHIRPS grid-reference file)
on a fresh deployment where it isn't already present locally - a deployed container starts from a git
checkout, and that data is deliberately gitignored (large, regenerable, doesn't belong in source
control). Local development is unaffected: if the files are already there (the normal case on a dev
machine that ran the data pipeline), this is a no-op.

Set GRAMCAST_DATA_BUNDLE_URL to a direct-download URL for gramcast_deploy_bundle.zip (see
scripts/prepare_deploy_bundle.py) - typically a GitHub Release asset URL.
"""
import os
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TRAINING_PAIRS_DIR = ROOT / "data" / "processed" / "training_pairs"
CHIRPS_DIR = ROOT / "data" / "raw" / "chirps"
CHECKPOINT_DIR = ROOT / "ml" / "checkpoints"
BOUNDARIES_DIR = ROOT / "data" / "boundaries"

REQUIRED_TRAINING_PAIR_FILES = [
    "fine_rainfall.npy",
    "coarse_rainfall_bicubic.npy",
    "terrain_static.npz",
    "valid_mask.npy",
    "dates.npy",
    "split_indices.npz",
    # 3 plain .npy files, not era5_humidity_wind.npz - inference.py mmaps these individually to avoid a
    # real OOM on Render's free tier; an .npz can't be mmap'd directly (see inference.py, scripts/
    # prepare_deploy_bundle.py). Filenames must match preprocessing.py's ERA5_CHANNELS.
    "era5_humidity_proxy.npy",
    "era5_wind_u.npy",
    "era5_wind_v.npy",
]
REQUIRED_CHIRPS_FILE = "chirps_pune_2023_07.nc"
# Real bug caught via a genuine CI failure, not proactively: this file is gitignored (data/ is a
# blanket-ignored directory) and was never actually included in prepare_deploy_bundle.py, so every
# deployed/CI boot's VillageIndex() (backend/main.py's lifespan startup, right after GramCastInference())
# failed with a FileNotFoundError before the app ever came up - the 41 "failing" tests in CI were this
# one root cause cascading, not 41 separate bugs.
REQUIRED_BOUNDARIES_FILE = "pune_villages.geojson"

DOWNLOAD_TIMEOUT_S = 300


def _is_data_present() -> bool:
    if not all((TRAINING_PAIRS_DIR / f).exists() for f in REQUIRED_TRAINING_PAIR_FILES):
        return False
    if not (CHIRPS_DIR / REQUIRED_CHIRPS_FILE).exists():
        return False
    if not (BOUNDARIES_DIR / REQUIRED_BOUNDARIES_FILE).exists():
        return False
    if not any(CHECKPOINT_DIR.glob("*_best.pt")):
        return False
    return True


def ensure_data_available() -> None:
    """Downloads and extracts the bundle if required files are missing AND a bundle URL is configured.
    Raises RuntimeError with a clear message if data is missing and no URL is set - fails loudly at
    startup rather than letting GramCastInference() fail later with a confusing FileNotFoundError deep
    in numpy/torch loading code."""
    if _is_data_present():
        return

    bundle_url = os.environ.get("GRAMCAST_DATA_BUNDLE_URL")
    if not bundle_url:
        raise RuntimeError(
            "Required data files are missing and GRAMCAST_DATA_BUNDLE_URL is not set. "
            "Either run the data pipeline locally, or set GRAMCAST_DATA_BUNDLE_URL to a "
            "gramcast_deploy_bundle.zip download URL (see scripts/prepare_deploy_bundle.py)."
        )

    print(f"Data files missing - downloading bundle from {bundle_url} ...")
    TRAINING_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    CHIRPS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = ROOT / "_gramcast_deploy_bundle_download.zip"
    with requests.get(bundle_url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.startswith("training_pairs/"):
                zf.extract(member, TRAINING_PAIRS_DIR.parent)
            elif member.startswith("chirps/"):
                zf.extract(member, CHIRPS_DIR.parent)
            elif member.startswith("checkpoints/"):
                zf.extract(member, CHECKPOINT_DIR.parent)
            elif member.startswith("boundaries/"):
                zf.extract(member, BOUNDARIES_DIR.parent)
    zip_path.unlink()

    if not _is_data_present():
        raise RuntimeError("Downloaded and extracted the bundle, but required files are still missing - check the bundle's contents match what prepare_deploy_bundle.py produced.")
    print("Data bundle downloaded and extracted successfully.")
