"""Ensemble evaluation: average P10/P50/P90 predictions across several independently-trained
checkpoints and compare the ensemble honestly against both the individual models and the 3 baselines.

Averaging is mathematically safe here for P50 specifically: every model's P50 is already forced by
its own mass-conservation layer to match the SAME coarse (nearest-neighbour) target block-mean - so
the average of N already-mass-conserving predictions still has that same block-mean exactly (the mean
of N copies of the same target value is that value), meaning the ensemble P50 is automatically still
mass-conserving too, not just an approximation. P10/P90 are simple averages (standard ensembling,
no equivalent guarantee needed there).

Usage: python ml/ensemble_evaluate.py --checkpoints ml/checkpoints/a_best.pt ml/checkpoints/b_best.pt ...
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
from preprocessing import ERA5_CHANNELS
from metrics import full_report
from nearest_neighbor import predict as nn_predict
from elevation_aware_regression import build_features
from sklearn.linear_model import LinearRegression

TRAINING_PAIRS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "training_pairs"


def get_predictions_for_checkpoint(path, train_ds, test_ds, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("args") or ckpt.get("config")
    full_channels = train_ds.input_tensor.shape[1]
    ckpt_in_channels = ckpt["model_state"]["enc1.block.0.weight"].shape[1]

    channel_indices = None
    if ckpt_in_channels != full_channels:
        if ckpt_in_channels == full_channels - len(ERA5_CHANNELS):
            channel_indices = [0] + list(range(1 + len(ERA5_CHANNELS), full_channels))
        else:
            raise ValueError(f"{path}: checkpoint expects {ckpt_in_channels} channels, unrecognized (current={full_channels})")

    model = ResidualUNet(in_channels=ckpt_in_channels, base_channels=ckpt_config["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    p10_all, p50_all, p90_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"]
            if channel_indices is not None:
                x = x[:, channel_indices]
            x = x.to(device)
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
    parser.add_argument("--checkpoints", nargs="+", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, stats, valid_mask = load_datasets()

    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    coarse_bicubic = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
    terrain = np.load(TRAINING_PAIRS_DIR / "terrain_static.npz")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    train_idx, test_idx = splits["train"], splits["test"]
    fine_test = fine[test_idx]

    print(f"Ensembling {len(args.checkpoints)} checkpoints:")
    per_model_preds = {}
    for ckpt_path in args.checkpoints:
        name = Path(ckpt_path).stem
        print(f"  {name}")
        per_model_preds[name] = get_predictions_for_checkpoint(ckpt_path, train_ds, test_ds, device)

    ensemble_p10 = np.mean([p["p10"] for p in per_model_preds.values()], axis=0)
    ensemble_p50 = np.mean([p["p50"] for p in per_model_preds.values()], axis=0)
    ensemble_p90 = np.mean([p["p90"] for p in per_model_preds.values()], axis=0)
    assert (ensemble_p10 <= ensemble_p50).all() and (ensemble_p50 <= ensemble_p90).all(), \
        "ensemble monotonicity broken - averaging should preserve p10<=p50<=p90 pointwise but didn't"

    baselines = {}
    baselines["nearest_neighbour"] = nn_predict(fine_test)
    baselines["bicubic"] = coarse_bicubic[test_idx]
    X_train = build_features(coarse_bicubic[train_idx], terrain)
    y_train = fine[train_idx].flatten()
    reg_model = LinearRegression().fit(X_train, y_train)
    X_test = build_features(coarse_bicubic[test_idx], terrain)
    baselines["elevation_aware_regression"] = np.clip(reg_model.predict(X_test), 0, None).reshape(len(test_idx), 35, 50)
    for name, preds in per_model_preds.items():
        baselines[f"unet_{name}"] = preds["p50"]
    baselines["unet_ENSEMBLE"] = ensemble_p50

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

    print("\nHonest summary: report every number above, including any metric where the ensemble does not win.")
    unet_names = [n for n in baselines if n.startswith("unet_")]
    best_rmse = min(unet_names, key=lambda k: reports[k]["rmse"])
    print(f"Best RMSE among U-Net variants: {best_rmse} ({reports[best_rmse]['rmse']:.4f})")
    ensemble_beats_all_individuals = all(
        reports["unet_ENSEMBLE"]["rmse"] <= reports[f"unet_{name}"]["rmse"] for name in per_model_preds
    )
    print(f"Ensemble RMSE ({reports['unet_ENSEMBLE']['rmse']:.4f}) beats every individual model: {ensemble_beats_all_individuals}")


if __name__ == "__main__":
    main()
