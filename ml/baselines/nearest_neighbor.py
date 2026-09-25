"""Nearest-neighbour baseline: the coarse (0.25deg) value copied to every fine cell inside it - what today's
block-level forecast actually gives a farmer, per the project guide's own framing ("what happens today").

Recomputes the coarse array via the same 5x5 block-average used in build_training_pairs.py (that block-mean
logic was already verified there - see DEV_LOG), then upsamples with plain repeat (nearest-neighbour), not
bicubic.
"""
from pathlib import Path

import numpy as np

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"
COARSEN_FACTOR = 5


def predict(fine_shape_grid: np.ndarray) -> np.ndarray:
    """fine_shape_grid: (T, 35, 50) fine rainfall - returns nearest-neighbour upsampled coarse prediction."""
    T, h, w = fine_shape_grid.shape
    ch, cw = h // COARSEN_FACTOR, w // COARSEN_FACTOR
    coarse = fine_shape_grid.reshape(T, ch, COARSEN_FACTOR, cw, COARSEN_FACTOR).mean(axis=(2, 4))
    nn_upsampled = np.repeat(np.repeat(coarse, COARSEN_FACTOR, axis=1), COARSEN_FACTOR, axis=2)
    return nn_upsampled


def main() -> None:
    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    valid = np.load(TRAINING_PAIRS_DIR / "valid_mask.npy")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import full_report

    test_idx = splits["test"]
    fine_test = fine[test_idx]
    pred_test = predict(fine_test)

    report = full_report(pred_test, fine_test, valid)
    print("Nearest-neighbour baseline, test set (2021-2026):")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
