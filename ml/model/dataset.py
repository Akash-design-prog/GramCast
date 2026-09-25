"""PyTorch Dataset wrapping the training pairs. Precomputes the nearest-neighbour coarse array once (needed as
the mass-conservation target - NOT the bicubic array, which doesn't itself conserve mass) rather than
recomputing it per-sample.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from nearest_neighbor import predict as nn_predict

from preprocessing import build_input_tensor

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "training_pairs"


class GramCastDataset(Dataset):
    def __init__(self, indices: np.ndarray, fine: np.ndarray, coarse_bicubic: np.ndarray, terrain: dict, stats: dict):
        self.indices = indices
        self.fine = fine[indices]
        self.coarse_bicubic = coarse_bicubic[indices]
        self.coarse_nn = nn_predict(self.fine)  # target for mass-conservation - recomputed from fine, matches build_training_pairs.py's own coarsening logic
        self.input_tensor = build_input_tensor(self.coarse_bicubic, terrain, stats)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return {
            "x": torch.from_numpy(self.input_tensor[i]),
            "bicubic_raw": torch.from_numpy(self.coarse_bicubic[i]).unsqueeze(0).float(),
            "coarse_nn_raw": torch.from_numpy(self.coarse_nn[i]).unsqueeze(0).float(),
            "truth": torch.from_numpy(self.fine[i]).unsqueeze(0).float(),
        }


def load_datasets():
    fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
    coarse_bicubic = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
    splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
    terrain = dict(np.load(TRAINING_PAIRS_DIR / "terrain_static.npz"))
    valid_mask = np.load(TRAINING_PAIRS_DIR / "valid_mask.npy")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocessing import compute_normalization_stats
    stats = compute_normalization_stats(terrain, coarse_bicubic[splits["train"]])

    train_ds = GramCastDataset(splits["train"], fine, coarse_bicubic, terrain, stats)
    val_ds = GramCastDataset(splits["val"], fine, coarse_bicubic, terrain, stats)
    test_ds = GramCastDataset(splits["test"], fine, coarse_bicubic, terrain, stats)
    return train_ds, val_ds, test_ds, stats, valid_mask
