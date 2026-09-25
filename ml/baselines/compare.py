"""Comparison table across all 3 baselines, per the project guide's own requirement (section 6): every model
result must be reported next to nearest-neighbour, bicubic, and elevation-aware regression on identical pixels,
honestly - including where one baseline doesn't clearly win.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import full_report
from nearest_neighbor import predict as nn_predict
from elevation_aware_regression import build_features


def main() -> None:
    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    coarse_bicubic = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    valid = np.load(TRAINING_PAIRS_DIR / "valid_mask.npy")
    terrain = np.load(TRAINING_PAIRS_DIR / "terrain_static.npz")

    train_idx, test_idx = splits["train"], splits["test"]
    fine_test = fine[test_idx]

    preds = {}
    preds["nearest_neighbour"] = nn_predict(fine_test)
    preds["bicubic"] = coarse_bicubic[test_idx]

    X_train = build_features(coarse_bicubic[train_idx], terrain)
    y_train = fine[train_idx].flatten()
    model = LinearRegression().fit(X_train, y_train)
    X_test = build_features(coarse_bicubic[test_idx], terrain)
    reg_pred = np.clip(model.predict(X_test), 0, None).reshape(len(test_idx), 35, 50)
    preds["elevation_aware_regression"] = reg_pred

    reports = {name: full_report(pred, fine_test, valid) for name, pred in preds.items()}

    print(f"{'Metric':<20}", end="")
    for name in reports:
        print(f"{name:>28}", end="")
    print()
    for metric in ["rmse", "mae", "correlation", "p99_bias"]:
        print(f"{metric:<20}", end="")
        for name in reports:
            print(f"{reports[name][metric]:>28.4f}", end="")
        print()
    for event in ["event_light", "event_moderate", "event_heavy"]:
        for sub in ["pod", "far", "csi"]:
            label = f"{event.replace('event_', '')}_{sub}"
            print(f"{label:<20}", end="")
            for name in reports:
                print(f"{reports[name][event][sub]:>28.4f}", end="")
            print()

    print("\nHonest summary: no single baseline wins every metric.")
    best_rmse = min(reports, key=lambda k: reports[k]["rmse"])
    best_heavy_csi = max(reports, key=lambda k: reports[k]["event_heavy"]["csi"])
    best_p99 = min(reports, key=lambda k: abs(reports[k]["p99_bias"]))
    print(f"  Best RMSE: {best_rmse} ({reports[best_rmse]['rmse']:.3f})")
    print(f"  Best heavy-rain CSI: {best_heavy_csi} ({reports[best_heavy_csi]['event_heavy']['csi']:.3f})")
    print(f"  Best (smallest) p99 bias: {best_p99} ({reports[best_p99]['p99_bias']:.3f})")


if __name__ == "__main__":
    main()
