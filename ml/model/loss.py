"""Loss function, per the project guide section 5: plain average error (MSE/MAE) flattens heavy-rain peaks,
which are exactly what farmers and disaster teams care about most. Train on log1p-transformed rainfall with
extra weight on heavy-rain cells, so the loss doesn't treat a 2mm error on a 5mm day the same as a 2mm error on
a 100mm day.

Weighting scheme: weight = 1 + HEAVY_WEIGHT * (truth >= HEAVY_THRESHOLD), i.e. cells at/above IMD's own heavy-
rain threshold (64.5 mm/day) get HEAVY_WEIGHT+1x the loss contribution of an ordinary cell. Applied on TRUTH,
not prediction - weighting by the prediction would let the network dodge the penalty by simply predicting low.
"""
import torch
import torch.nn as nn

HEAVY_THRESHOLD_MM = 64.5  # IMD's own heavy-rain category threshold, same as metrics.py's event scoring
HEAVY_WEIGHT = 4.0  # heavy-rain cells count for 5x an ordinary cell's loss contribution


class HeavyRainWeightedLoss(nn.Module):
    def __init__(self, heavy_threshold: float = HEAVY_THRESHOLD_MM, heavy_weight: float = HEAVY_WEIGHT):
        super().__init__()
        self.heavy_threshold = heavy_threshold
        self.heavy_weight = heavy_weight

    def forward(self, pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """pred, truth: (B, 1, H, W) raw (unlogged, non-negative) rainfall. mask: (H, W) bool, True = valid
        cell - excludes the known static coastal-artifact cells, same discipline as metrics.py.
        """
        log_pred = torch.log1p(pred)
        log_truth = torch.log1p(truth)

        weight = 1.0 + self.heavy_weight * (truth >= self.heavy_threshold).float()

        sq_err = (log_pred - log_truth) ** 2
        weighted_sq_err = sq_err * weight

        if mask is not None:
            weighted_sq_err = weighted_sq_err[:, :, mask]
            return weighted_sq_err.mean()
        return weighted_sq_err.mean()
