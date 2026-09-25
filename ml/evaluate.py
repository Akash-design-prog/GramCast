"""Full evaluation: load a trained checkpoint, run it against the real held-out test set, report honestly
against the 3 baselines - per the project guide's section 6 requirement that every result be shown next to
nearest-neighbour, bicubic, and elevation-aware regression on identical pixels, even where the model doesn't
clearly win.

Also checks P10/P90 coverage - the guide's own uncertainty requirement isn't satisfied just by producing three
numbers per cell; the numbers need to actually mean something. A well-calibrated P10-P90 band should contain
the true value roughly 80% of the time (since P10 = the 10th percentile, P90 = the 90th, 90-10=80). Coverage
far from 80% means the quantile heads are outputting three numbers that only look like quantiles.

Usage: python ml/evaluate.py --checkpoint ml/checkpoints/<run>_best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "baselines"))
from dataset import load_datasets
from unet import ResidualUNet
from metrics import full_report
from nearest_neighbor import predict as nn_predict
from elevation_aware_regression import build_features
from sklearn.linear_model import LinearRegression

TRAINING_PAIRS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "training_pairs"


def coverage_check(p10: np.ndarray, p90: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    """What fraction of true values actually fall within [p10, p90]? Should be close to 80% if the quantile
    heads are genuinely calibrated, not just three numbers that happen to be ordered p10<=p50<=p90."""
    in_band = (truth >= p10) & (truth <= p90)
    coverage = float(in_band[:, mask].mean())
    below_p10 = float((truth < p10)[:, mask].mean())
    above_p90 = float((truth > p90)[:, mask].mean())
    return {"coverage_80_target": coverage, "below_p10_frac": below_p10, "above_p90_frac": above_p90}


def get_unet_predictions(model, ds, device) -> dict:
    """Run the model over the full dataset in batches, return {p10, p50, p90} as (T,35,50) numpy arrays."""
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=64, shuffle=False)
    p10_all, p50_all, p90_all = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            bicubic_raw = batch["bicubic_raw"].to(device)
            coarse_nn_raw = batch["coarse_nn_raw"].to(device)
            q = model.predict_quantiles(x, bicubic_raw, coarse_nn_raw)
            p10_all.append(q["p10"].cpu().numpy())
            p50_all.append(q["p50"].cpu().numpy())
            p90_all.append(q["p90"].cpu().numpy())
    return {
        "p10": np.concatenate(p10_all)[:, 0],
        "p50": np.concatenate(p50_all)[:, 0],
        "p90": np.concatenate(p90_all)[:, 0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")

    train_ds, val_ds, test_ds, stats, valid_mask = load_datasets()
    model = ResidualUNet(in_channels=train_ds.input_tensor.shape[1], base_channels=ckpt["args"]["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    print("Running U-Net over the real test set...")
    unet_preds = get_unet_predictions(model, test_ds, device)

    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    coarse_bicubic = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    terrain = np.load(TRAINING_PAIRS_DIR / "terrain_static.npz")
    train_idx, test_idx = splits["train"], splits["test"]
    fine_test = fine[test_idx]

    baselines = {}
    baselines["nearest_neighbour"] = nn_predict(fine_test)
    baselines["bicubic"] = coarse_bicubic[test_idx]
    X_train = build_features(coarse_bicubic[train_idx], terrain)
    y_train = fine[train_idx].flatten()
    reg_model = LinearRegression().fit(X_train, y_train)
    X_test = build_features(coarse_bicubic[test_idx], terrain)
    baselines["elevation_aware_regression"] = np.clip(reg_model.predict(X_test), 0, None).reshape(len(test_idx), 35, 50)
    baselines["unet_p50"] = unet_preds["p50"]

    print(f"\n{'Metric':<20}", end="")
    for name in baselines:
        print(f"{name:>28}", end="")
    print()
    reports = {name: full_report(pred, fine_test, valid_mask) for name, pred in baselines.items()}
    for metric in ["rmse", "mae", "correlation", "p99_bias"]:
        print(f"{metric:<20}", end="")
        for name in baselines:
            print(f"{reports[name][metric]:>28.4f}", end="")
        print()
    for event in ["event_light", "event_moderate", "event_heavy"]:
        for sub in ["pod", "far", "csi"]:
            label = f"{event.replace('event_', '')}_{sub}"
            print(f"{label:<20}", end="")
            for name in baselines:
                print(f"{reports[name][event][sub]:>28.4f}", end="")
            print()

    print("\n--- U-Net P10/P90 uncertainty calibration ---")
    cov = coverage_check(unet_preds["p10"], unet_preds["p90"], fine_test, valid_mask)
    print(f"Coverage (fraction of true values inside [P10,P90], target ~0.80): {cov['coverage_80_target']:.4f}")
    print(f"Fraction below P10 (target ~0.10): {cov['below_p10_frac']:.4f}")
    print(f"Fraction above P90 (target ~0.10): {cov['above_p90_frac']:.4f}")
    if abs(cov["coverage_80_target"] - 0.80) > 0.15:
        print("WARNING: coverage is far from the 80% target - the quantile heads may not be well-calibrated yet.")
        print("Consider split-conformal calibration (see docs/ISSUES_PLAN.md) before trusting the uncertainty band.")

    print("\nHonest summary: report every number above, including any metric where the U-Net does not win.")
    best_rmse = min(baselines, key=lambda k: reports[k]["rmse"])
    best_heavy_csi = max(baselines, key=lambda k: reports[k]["event_heavy"]["csi"])
    print(f"Best RMSE: {best_rmse} ({reports[best_rmse]['rmse']:.4f})")
    print(f"Best heavy-rain CSI: {best_heavy_csi} ({reports[best_heavy_csi]['event_heavy']['csi']:.4f})")


if __name__ == "__main__":
    main()
