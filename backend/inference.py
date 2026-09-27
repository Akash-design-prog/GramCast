"""Model inference wrapper: loads a trained checkpoint once and the pre-built, pre-verified training-pair
arrays once, then serves P10/P50/P90 predictions for any date in `dates.npy` on demand.

Scope note (deliberate, not an oversight): this serves HISTORICAL days from the already-built, already-
verified `data/processed/training_pairs/` arrays (CHIRPS-derived), not a live GFS forecast - that's the
guide's own "past-event mode vs live-forecast mode" distinction (section 5b), and live-forecast mode is a
separate frontend-listed item (`docs/ISSUES_PLAN.md`), not something this wrapper needs to solve today.

Reuses the exact checkpoint-channel-count detection already verified in ml/evaluate.py (a checkpoint
trained before ERA5 was wired in expects 14 input channels; today's dataset build produces 17) - kept as
a small self-contained copy here rather than importing ml/evaluate.py, since that module's main() has
side effects (argparse) this wrapper shouldn't trigger.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_ML_MODEL_DIR = Path(__file__).resolve().parents[1] / "ml" / "model"
sys.path.insert(0, str(_ML_MODEL_DIR))
from unet import ResidualUNet  # noqa: E402
from preprocessing import ERA5_CHANNELS, build_input_tensor, compute_normalization_stats  # noqa: E402

TRAINING_PAIRS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "training_pairs"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "ml" / "checkpoints" / "gramcast_20260926_074738_heavy_weight_8_best.pt"  # best real checkpoint so far, ERA5-wired (see DEV_LOG 2026-09-26)


class GramCastInference:
    """Loads everything once at construction time (meant to be a FastAPI app-lifetime singleton, not
    re-instantiated per request - the checkpoint and the full 5,582-day arrays are not cheap to reload)."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fine = np.load(TRAINING_PAIRS_DIR / "fine_rainfall.npy")
        self.coarse_bicubic = np.load(TRAINING_PAIRS_DIR / "coarse_rainfall_bicubic.npy")
        self.dates = np.load(TRAINING_PAIRS_DIR / "dates.npy")
        self.valid_mask = np.load(TRAINING_PAIRS_DIR / "valid_mask.npy")
        self.terrain = dict(np.load(TRAINING_PAIRS_DIR / "terrain_static.npz"))
        era5_npz = np.load(TRAINING_PAIRS_DIR / "era5_humidity_wind.npz")
        self.era5 = {name: era5_npz[name] for name in ERA5_CHANNELS}

        splits = np.load(TRAINING_PAIRS_DIR / "split_indices.npz")
        train_idx = splits["train"]
        era5_train = {name: arr[train_idx] for name, arr in self.era5.items()}
        self.stats = compute_normalization_stats(self.terrain, self.coarse_bicubic[train_idx], era5_train)

        self._date_to_idx = {str(d): i for i, d in enumerate(self.dates)}
        self._sorted_dates = sorted(self._date_to_idx.keys())

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_config = ckpt.get("args") or ckpt.get("config")
        if ckpt_config is None:
            raise KeyError(f"{checkpoint_path}: checkpoint has neither 'args' nor 'config' key")

        full_channels = 1 + len(ERA5_CHANNELS) + 4 + 9  # rain + era5 + (dem,slope,sin,cos) + worldcover
        ckpt_in_channels = ckpt["model_state"]["enc1.block.0.weight"].shape[1]
        self.channel_indices = None
        if ckpt_in_channels != full_channels:
            if ckpt_in_channels == full_channels - len(ERA5_CHANNELS):
                self.channel_indices = [0] + list(range(1 + len(ERA5_CHANNELS), full_channels))
            else:
                raise ValueError(f"{checkpoint_path}: expects {ckpt_in_channels} channels, unrecognized layout")

        self.model = ResidualUNet(in_channels=ckpt_in_channels, base_channels=ckpt_config["base_channels"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_epoch = ckpt["epoch"]

    def available_dates(self) -> list[str]:
        return list(self._date_to_idx.keys())

    def sorted_available_dates(self) -> list[str]:
        """Same dates as available_dates(), pre-sorted once at load time - the /dates endpoint (and
        anything else that wants a real chronological list, e.g. the frontend date picker) doesn't need
        to re-sort all 5,582 strings on every single call."""
        return self._sorted_dates

    def predict_day(self, date_str: str) -> dict:
        """Returns a dict of (35, 50) south-ascending float32 arrays: p10, p50, p90, block_value
        (the flat nearest-neighbour block-level value the guide contrasts against panchayat-level output),
        plus the requested date and this wrapper's checkpoint metadata. Raises ValueError for a date not
        present in dates.npy (rather than silently returning something for the nearest date)."""
        if date_str not in self._date_to_idx:
            raise ValueError(f"'{date_str}' not found in this dataset's dates.npy - no prediction available for that day")
        idx = self._date_to_idx[date_str]

        coarse_bicubic_day = self.coarse_bicubic[idx : idx + 1]
        era5_day = {name: arr[idx : idx + 1] for name, arr in self.era5.items()}
        x = build_input_tensor(coarse_bicubic_day, self.terrain, self.stats, era5_day)
        if self.channel_indices is not None:
            x = x[:, self.channel_indices]

        fine_day = self.fine[idx : idx + 1]
        h, w = fine_day.shape[1], fine_day.shape[2]
        ch, cw = h // 5, w // 5
        coarse_nn = fine_day.reshape(1, ch, 5, cw, 5).mean(axis=(2, 4))
        block_value = np.repeat(np.repeat(coarse_nn, 5, axis=1), 5, axis=2)

        x_t = torch.from_numpy(x).to(self.device)
        bicubic_t = torch.from_numpy(coarse_bicubic_day).unsqueeze(1).float().to(self.device)
        coarse_nn_t = torch.from_numpy(block_value).unsqueeze(1).float().to(self.device)

        with torch.no_grad():
            q = self.model.predict_quantiles(x_t, bicubic_t, coarse_nn_t)

        return {
            "date": date_str,
            "p10": q["p10"][0, 0].cpu().numpy(),
            "p50": q["p50"][0, 0].cpu().numpy(),
            "p90": q["p90"][0, 0].cpu().numpy(),
            "block_value": block_value[0],
            "checkpoint": str(self.checkpoint_path.name),
            "checkpoint_epoch": self.checkpoint_epoch,
        }
