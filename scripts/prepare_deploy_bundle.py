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
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TRAINING_PAIRS_DIR = ROOT / "data" / "processed" / "training_pairs"
CHIRPS_SAMPLE = ROOT / "data" / "raw" / "chirps" / "chirps_pune_2023_07.nc"
BEST_CHECKPOINT = ROOT / "ml" / "checkpoints" / "gramcast_20260926_074738_heavy_weight_8_best.pt"
OUT_PATH = ROOT / "gramcast_deploy_bundle.zip"

TRAINING_PAIR_FILES = [
    "fine_rainfall.npy",
    "coarse_rainfall_bicubic.npy",
    "terrain_static.npz",
    "valid_mask.npy",
    "dates.npy",
    "split_indices.npz",
    "era5_humidity_wind.npz",
]


def main() -> None:
    missing = [TRAINING_PAIRS_DIR / f for f in TRAINING_PAIR_FILES if not (TRAINING_PAIRS_DIR / f).exists()]
    missing += [p for p in [CHIRPS_SAMPLE, BEST_CHECKPOINT] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s), cannot build bundle: {missing}")

    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in TRAINING_PAIR_FILES:
            zf.write(TRAINING_PAIRS_DIR / name, arcname=f"training_pairs/{name}")
        zf.write(CHIRPS_SAMPLE, arcname="chirps/chirps_pune_2023_07.nc")
        zf.write(BEST_CHECKPOINT, arcname=f"checkpoints/{BEST_CHECKPOINT.name}")

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
