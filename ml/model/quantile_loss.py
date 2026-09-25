"""Pinball (quantile) loss for the P10/P50/P90 heads, combined with the same log1p transform and heavy-rain
weighting already used for the point-estimate loss (see loss.py) - the same reasoning applies: plain error
metrics flatten heavy-rain peaks, and that applies to quantile predictions just as much as point ones.

Standard pinball loss for quantile tau: L_tau(y, y_hat) = max(tau*(y-y_hat), (tau-1)*(y-y_hat))
"""
import torch
import torch.nn as nn

HEAVY_THRESHOLD_MM = 64.5
HEAVY_WEIGHT = 4.0
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}


def pinball_loss(pred: torch.Tensor, truth: torch.Tensor, tau: float) -> torch.Tensor:
    diff = truth - pred
    return torch.maximum(tau * diff, (tau - 1) * diff)


class QuantileHeavyRainLoss(nn.Module):
    def __init__(self, heavy_threshold: float = HEAVY_THRESHOLD_MM, heavy_weight: float = HEAVY_WEIGHT):
        super().__init__()
        self.heavy_threshold = heavy_threshold
        self.heavy_weight = heavy_weight

    def forward(self, preds: dict, truth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """preds: {"p10":..., "p50":..., "p90":...}, each (B,1,H,W) raw non-negative rainfall.
        truth: (B,1,H,W). mask: (H,W) bool."""
        log_truth = torch.log1p(truth)
        weight = 1.0 + self.heavy_weight * (truth >= self.heavy_threshold).float()

        total = 0.0
        for name, tau in QUANTILES.items():
            log_pred = torch.log1p(preds[name])
            pin = pinball_loss(log_pred, log_truth, tau) * weight
            if mask is not None:
                pin = pin[:, :, mask]
            total = total + pin.mean()
        return total / len(QUANTILES)
