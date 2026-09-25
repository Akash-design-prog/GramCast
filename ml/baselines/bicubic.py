"""Bicubic baseline: the already-built coarse_rainfall_bicubic.npy IS this baseline (see
build_training_pairs.py) - just evaluate it against fine_rainfall.npy on the held-out test set.
"""
from pathlib import Path
import sys

import numpy as np

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"


def main() -> None:
    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    coarse_bicubic = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    valid = np.load(TRAINING_PAIRS_DIR / "valid_mask.npy")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import full_report

    test_idx = splits["test"]
    fine_test = fine[test_idx]
    pred_test = coarse_bicubic[test_idx]

    report = full_report(pred_test, fine_test, valid)
    print("Bicubic baseline, test set (2021-2026):")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
