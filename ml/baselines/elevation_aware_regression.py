"""Elevation-aware regression baseline: linear regression of fine rainfall on [bicubic-upsampled coarse value,
elevation, slope] - the sensible non-deep-learning method the project guide asks to beat (section 6).

Fit ONLY on the train split (1981-2018), never on val/test - the same discipline enforced throughout this
pipeline for the deep model.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"


def build_features(coarse_bicubic: np.ndarray, terrain: dict) -> np.ndarray:
    """coarse_bicubic: (T, 35, 50). Returns (T*35*50, 3) feature matrix: [bicubic_value, elevation, slope]."""
    T = coarse_bicubic.shape[0]
    dem_tiled = np.tile(terrain["dem"], (T, 1, 1))
    slope_tiled = np.tile(terrain["slope"], (T, 1, 1))
    features = np.stack([coarse_bicubic.flatten(), dem_tiled.flatten(), slope_tiled.flatten()], axis=1)
    return features


def main() -> None:
    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    coarse_bicubic = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    valid = np.load(TRAINING_PAIRS_DIR / "valid_mask.npy")
    terrain = np.load(TRAINING_PAIRS_DIR / "terrain_static.npz")

    train_idx, test_idx = splits["train"], splits["test"]

    X_train = build_features(coarse_bicubic[train_idx], terrain)
    y_train = fine[train_idx].flatten()

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Fitted on {len(y_train)} train pixels. Coefficients: {dict(zip(['bicubic', 'elevation', 'slope'], model.coef_))}, intercept: {model.intercept_}")

    X_test = build_features(coarse_bicubic[test_idx], terrain)
    y_pred_flat = model.predict(X_test)
    y_pred_flat = np.clip(y_pred_flat, 0, None)  # rain can't be negative - same discipline as the bicubic fix
    pred_test = y_pred_flat.reshape(len(test_idx), 35, 50)
    fine_test = fine[test_idx]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import full_report

    report = full_report(pred_test, fine_test, valid)
    print("\nElevation-aware regression baseline, test set (2021-2026):")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
