"""Split-conformal calibration of the P10/P90 band, adapted from GeoEnhance-AI/ml/conformal.py's
methodology (Conformalized Quantile Regression, Romano et al.) for this project's quantile U-Net.

Why this is needed: ml/evaluate.py's honest coverage check on the real checkpoint measured 91.9% of
true values falling inside [P10, P90] against an 80% target - the band is too WIDE (under-confident),
not miscalibrated-narrow. That number alone doesn't fix anything; this module computes a single,
data-driven correction and re-checks coverage on a genuinely held-out set.

Disjoint fit/calibrate/test split (same discipline as GeoEnhance-AI's conformal.py, adapted to this
project's own splits, which are already disjoint BY TIME - no new split needed):
- The model was FIT (trained) on the train split (1981-2018).
- The calibration set is the VAL split (2019-2020) - never touched during training, and disjoint in
  time from both train and test.
- Coverage is checked ONLY on the TEST split (2021-2026) - never touched during training or
  calibration. This is the one honest place to report "did the calibrated band actually cover 80%
  of held-out reality."

Nonconformity score (standard CQR): for each valid pixel-day, e_i = max(p10_i - y_i, y_i - p90_i) -
positive when the true value falls outside the predicted band (how far outside), negative/zero when
inside. q_hat is the finite-sample-correct (1-alpha) quantile of these scores on the calibration set.
The calibrated band is then [p10 - q_hat, p90 + q_hat] (clipped at 0, since rainfall can't be
negative) - a single global correction, not per-pixel-adaptive (our band's width already varies
per-pixel via the model's own softplus deltas, unlike GeoEnhance-AI's flat regression-line case where
per-tile adaptivity had to be added separately).

Usage: python ml/conformal.py --checkpoint ml/checkpoints/<run>_best.pt
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from dataset import load_datasets
from unet import ResidualUNet

ALPHA = 0.20  # target 80% coverage (P10-P90), matching the guide's own P10/P90 band definition
EPS = 1e-6


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The standard finite-sample-correct split-conformal quantile: the ceil((n+1)(1-alpha))/n
    empirical quantile of the calibration nonconformity scores, clipped at 1 (asking for a quantile
    beyond the data means 'the interval must cover everything seen so far')."""
    n = scores.size
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def nonconformity_scores(p10: np.ndarray, p90: np.ndarray, truth: np.ndarray, mask: np.ndarray, wet_only: bool = False) -> np.ndarray:
    if wet_only:
        combined_mask = (truth > 0) & mask[np.newaxis, :, :]
        return np.maximum(p10 - truth, truth - p90)[combined_mask]
    p10_v, p90_v, truth_v = p10[:, mask], p90[:, mask], truth[:, mask]
    return np.maximum(p10_v - truth_v, truth_v - p90_v).flatten()


def coverage(p10: np.ndarray, p90: np.ndarray, truth: np.ndarray, mask: np.ndarray, wet_only: bool = False) -> dict:
    """wet_only=True restricts to pixel-days where truth > 0 - rainfall is heavily zero-inflated, and a
    correctly-predicted dry day (p10=truth=0) is a trivial, uninformative 'covered' case that dilutes
    the aggregate coverage number away from where calibration actually matters for a farmer."""
    if wet_only:
        wet = truth > 0
        combined_mask = wet & mask[np.newaxis, :, :]
        in_band = (truth >= p10) & (truth <= p90)
        n = combined_mask.sum()
        return {
            "coverage": float(in_band[combined_mask].sum() / n) if n else float("nan"),
            "below_p10_frac": float((truth < p10)[combined_mask].sum() / n) if n else float("nan"),
            "above_p90_frac": float((truth > p90)[combined_mask].sum() / n) if n else float("nan"),
            "mean_half_width": float(((p90 - p10) / 2)[combined_mask].mean()) if n else float("nan"),
            "n_pixel_days": int(n),
        }
    in_band = (truth >= p10) & (truth <= p90)
    return {
        "coverage": float(in_band[:, mask].mean()),
        "below_p10_frac": float((truth < p10)[:, mask].mean()),
        "above_p90_frac": float((truth > p90)[:, mask].mean()),
        "mean_half_width": float(((p90 - p10) / 2)[:, mask].mean()),
    }


def get_predictions(model, ds, device) -> dict:
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
    ckpt_config = ckpt.get("args") or ckpt.get("config")
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")

    train_ds, val_ds, test_ds, stats, valid_mask = load_datasets()
    full_channels = train_ds.input_tensor.shape[1]
    ckpt_in_channels = ckpt["model_state"]["enc1.block.0.weight"].shape[1]
    channel_indices = None
    if ckpt_in_channels != full_channels:
        from preprocessing import ERA5_CHANNELS
        if ckpt_in_channels == full_channels - len(ERA5_CHANNELS):
            print(f"NOTE: pre-ERA5 checkpoint ({ckpt_in_channels} channels) - evaluating on the matching channel subset.")
            channel_indices = [0] + list(range(1 + len(ERA5_CHANNELS), full_channels))
        else:
            raise ValueError(f"checkpoint expects {ckpt_in_channels} channels, unrecognized layout (current={full_channels})")

    model = ResidualUNet(in_channels=ckpt_in_channels, base_channels=ckpt_config["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    def predict(ds):
        if channel_indices is None:
            return get_predictions(model, ds, device)
        # reuse get_predictions but slice x first - small inline wrapper dataset
        class _Sliced(torch.utils.data.Dataset):
            def __init__(self, base):
                self.base = base

            def __len__(self):
                return len(self.base)

            def __getitem__(self, i):
                item = dict(self.base[i])
                item["x"] = item["x"][channel_indices]
                return item

        return get_predictions(model, _Sliced(ds), device)

    print("Computing calibration-set (val, 2019-2020) predictions...")
    cal_preds = predict(val_ds)
    fine = np.load(Path(__file__).resolve().parent.parent / "data" / "processed" / "training_pairs" / "fine_rainfall.npy")
    splits = np.load(Path(__file__).resolve().parent.parent / "data" / "processed" / "training_pairs" / "split_indices.npz")
    fine_val = fine[splits["val"]]
    fine_test = fine[splits["test"]]

    print("\nComputing test-set (2021-2026) predictions...")
    test_preds = predict(test_ds)

    for label, wet_only in [("ALL DAYS (incl. correctly-predicted dry days)", False), ("WET DAYS ONLY (truth > 0mm)", True)]:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        scores = nonconformity_scores(cal_preds["p10"], cal_preds["p90"], fine_val, valid_mask, wet_only=wet_only)
        q_hat = conformal_quantile(scores, ALPHA)
        print(f"Calibration set: {scores.size} pixel-days, q_hat (band correction): {q_hat:.4f} mm")

        before = coverage(test_preds["p10"], test_preds["p90"], fine_test, valid_mask, wet_only=wet_only)
        p10_cal = np.clip(test_preds["p10"] - q_hat, 0.0, None)
        p90_cal = test_preds["p90"] + q_hat
        after = coverage(p10_cal, p90_cal, fine_test, valid_mask, wet_only=wet_only)

        print(f"{'Metric':<25}{'Before calibration':>20}{'After calibration':>20}{'Target':>10}")
        print(f"{'Coverage (P10-P90)':<25}{before['coverage']:>20.4f}{after['coverage']:>20.4f}{1 - ALPHA:>10.2f}")
        print(f"{'Below P10 frac':<25}{before['below_p10_frac']:>20.4f}{after['below_p10_frac']:>20.4f}{ALPHA / 2:>10.2f}")
        print(f"{'Above P90 frac':<25}{before['above_p90_frac']:>20.4f}{after['above_p90_frac']:>20.4f}{ALPHA / 2:>10.2f}")
        print(f"{'Mean half-width (mm)':<25}{before['mean_half_width']:>20.4f}{after['mean_half_width']:>20.4f}{'-':>10}")

        if abs(after["coverage"] - (1 - ALPHA)) < abs(before["coverage"] - (1 - ALPHA)):
            print(f"Calibration improved coverage: {before['coverage']:.1%} -> {after['coverage']:.1%} (target {1 - ALPHA:.0%}).")
        else:
            print("Calibration did NOT improve coverage here - reported as-is, not claimed as a fix.")


if __name__ == "__main__":
    main()
