"""Builds gramcast_deploy_bundle.zip: everything the deployed backend needs at runtime that isn't
already in git (training-pair arrays, the best checkpoint, and one small CHIRPS sample file the
zonal-stats grid transform reads for its coordinate reference). Meant to be uploaded as a GitHub
Release asset (2GB/file limit, well above this bundle's size) - the deployed service downloads and
extracts it once at startup (see backend/bootstrap_data.py), so this data never needs to live in git.

Usage: python scripts/prepare_deploy_bundle.py
Then (you run this yourself, per the standing convention - I don't push/publish on your behalf):
  gh release create deploy-data-v1 gramcast_deploy_bundle.zip --repo Akash-design-prog/GramCast --title "Deploy data bundle v1" --notes "Training-pair arrays + best checkpoint for the deployed API"
"""
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "model"))
from preprocessing import ERA5_CHANNELS  # noqa: E402

TRAINING_PAIRS_DIR = ROOT / "data" / "processed" / "training_pairs"
CHIRPS_SAMPLE = ROOT / "data" / "raw" / "chirps" / "chirps_pune_2023_07.nc"
BEST_CHECKPOINT = ROOT / "ml" / "checkpoints" / "gramcast_20260926_074738_heavy_weight_8_best.pt"
# Real bug found via a genuine CI failure: this file is gitignored and was missing from every bundle
# built before this fix, so VillageIndex() (backend/villages.py, loaded at FastAPI startup right after
# GramCastInference()) failed with FileNotFoundError on every deployed/CI boot - see bootstrap_data.py's
# own comment on REQUIRED_BOUNDARIES_FILE for the full story.
VILLAGE_BOUNDARIES = ROOT / "data" / "boundaries" / "pune_villages.geojson"
OUT_PATH = ROOT / "gramcast_deploy_bundle.zip"

# era5_humidity_wind.npz itself is NOT in this list / NOT bundled: it's a zip container, which numpy
# cannot mmap_mode="r" into directly (compressed or not) - backend/inference.py needs flat, individually
# mmap-able .npy files instead, to avoid a real OOM on Render's free 512MB tier (see inference.py's own
# comment). The 3 files below are generated fresh from the real npz each time this script runs, so the
# npz itself stays the single source of truth for the training pipeline (which still reads it directly).
ERA5_SPLIT_FILES = [f"era5_{name}.npy" for name in ERA5_CHANNELS]

TRAINING_PAIR_FILES = [
    "fine_rainfall.npy",
    "coarse_rainfall_bicubic.npy",
    "terrain_static.npz",
    "valid_mask.npy",
    "dates.npy",
    "split_indices.npz",
    *ERA5_SPLIT_FILES,
]


def _ensure_era5_split_files() -> None:
    npz_path = TRAINING_PAIRS_DIR / "era5_humidity_wind.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}, cannot derive the era5 split files")
    npz = np.load(npz_path)
    for name in ERA5_CHANNELS:
        np.save(TRAINING_PAIRS_DIR / f"era5_{name}.npy", npz[name])


def main() -> None:
    _ensure_era5_split_files()
    missing = [TRAINING_PAIRS_DIR / f for f in TRAINING_PAIR_FILES if not (TRAINING_PAIRS_DIR / f).exists()]
    missing += [p for p in [CHIRPS_SAMPLE, BEST_CHECKPOINT, VILLAGE_BOUNDARIES] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s), cannot build bundle: {missing}")

    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in TRAINING_PAIR_FILES:
            zf.write(TRAINING_PAIRS_DIR / name, arcname=f"training_pairs/{name}")
        zf.write(CHIRPS_SAMPLE, arcname="chirps/chirps_pune_2023_07.nc")
        zf.write(BEST_CHECKPOINT, arcname=f"checkpoints/{BEST_CHECKPOINT.name}")
        zf.write(VILLAGE_BOUNDARIES, arcname=f"boundaries/{VILLAGE_BOUNDARIES.name}")

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")
    print("\nNext step (run yourself):")
    print(
        f'  gh release create deploy-data-v1 "{OUT_PATH}" --repo Akash-design-prog/GramCast '
        '--title "Deploy data bundle v1" --notes "Training-pair arrays + best checkpoint for the deployed API"'
    )
    print("\nThen copy the release asset's download URL into Render's GRAMCAST_DATA_BUNDLE_URL env var.")


if __name__ == "__main__":
    main()
