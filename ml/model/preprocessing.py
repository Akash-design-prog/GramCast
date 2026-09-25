"""Builds the U-Net input tensor: bicubic rainfall + terrain + land-cover channels, properly encoded.

Aspect is circular (0deg==360deg) - feeding raw degrees to a network would teach it that 359deg and 1deg are
maximally different, when they're actually almost the same direction. This is the same lesson already learned
the hard way when aspect was naively averaged during resampling (see DEV_LOG 2026-09-25) - decompose into
sin/cos here too, not just during resampling.

WorldCover is categorical (land cover class codes) - one-hot encoded, not fed as a raw number (40 is not
"between" 30 and 50 in any meaningful sense for a network to learn from a single continuous channel).

Normalization stats (mean/std for continuous channels) are computed from the TRAIN split ONLY, never touching
val/test - the same leakage discipline enforced throughout this pipeline.
"""
from pathlib import Path

import numpy as np

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"

# WorldCover classes actually present in this bbox (confirmed in DEV_LOG's WorldCover verification)
WORLDCOVER_CLASSES = [10, 20, 30, 40, 50, 60, 80, 90, 95]


def compute_normalization_stats(terrain: dict, coarse_bicubic_train: np.ndarray) -> dict:
    """Stats from TRAIN data only."""
    return {
        "dem_mean": float(terrain["dem"].mean()), "dem_std": float(terrain["dem"].std()),
        "slope_mean": float(terrain["slope"].mean()), "slope_std": float(terrain["slope"].std()),
        "rain_mean": float(coarse_bicubic_train.mean()), "rain_std": float(coarse_bicubic_train.std()),
    }


def build_input_tensor(coarse_bicubic: np.ndarray, terrain: dict, stats: dict) -> np.ndarray:
    """coarse_bicubic: (T, H, W). terrain arrays: (H, W), time-invariant, tiled across T.
    Returns (T, C, H, W) float32 - channel order: [rain, dem, slope, aspect_sin, aspect_cos, *worldcover_onehot]
    """
    T, H, W = coarse_bicubic.shape
    rain_norm = (coarse_bicubic - stats["rain_mean"]) / stats["rain_std"]
    dem_norm = (terrain["dem"] - stats["dem_mean"]) / stats["dem_std"]
    slope_norm = (terrain["slope"] - stats["slope_mean"]) / stats["slope_std"]
    aspect_rad = np.deg2rad(terrain["aspect"])
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    unknown_mask = ~np.isin(terrain["worldcover"], WORLDCOVER_CLASSES)
    if unknown_mask.any():
        unknown_classes = sorted(set(terrain["worldcover"][unknown_mask].tolist()))
        raise ValueError(
            f"WorldCover class(es) {unknown_classes} not in WORLDCOVER_CLASSES {WORLDCOVER_CLASSES} - "
            "these pixels would silently get an all-zero one-hot (invisible to the network) if not caught here. "
            "Add the missing class(es) to WORLDCOVER_CLASSES."
        )

    static_channels = [dem_norm, slope_norm, aspect_sin, aspect_cos]
    for cls in WORLDCOVER_CLASSES:
        static_channels.append((terrain["worldcover"] == cls).astype("float32"))

    n_channels = 1 + len(static_channels)
    tensor = np.empty((T, n_channels, H, W), dtype="float32")
    tensor[:, 0, :, :] = rain_norm
    for i, ch in enumerate(static_channels):
        tensor[:, i + 1, :, :] = np.tile(ch, (T, 1, 1))
    return tensor


def channel_names() -> list[str]:
    return ["rain_bicubic", "dem", "slope", "aspect_sin", "aspect_cos"] + [f"worldcover_{c}" for c in WORLDCOVER_CLASSES]


if __name__ == "__main__":
    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    coarse = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    terrain = np.load(TRAINING_PAIRS_DIR / "terrain_static.npz")

    stats = compute_normalization_stats(terrain, coarse[splits["train"]])
    print("Normalization stats (train-only):", stats)

    tensor = build_input_tensor(coarse[:5], terrain, stats)
    print(f"Input tensor shape: {tensor.shape} (expect (5, {1+3+len(WORLDCOVER_CLASSES)}, 35, 50))")
    print(f"Channel names: {channel_names()}")
    print(f"NaN in tensor: {np.isnan(tensor).sum()} (expect 0)")
