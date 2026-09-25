"""Mass-conservation layer, per the project guide section 5: after the network predicts a fine-resolution
field, rescale it so the average of the fine cells inside each coarse (0.25deg, 5x5) cell equals the original
coarse value - the fine field can only redistribute the rain the coarse input already said would fall, never
invent or delete it.
"""
import torch
import torch.nn as nn

COARSEN_FACTOR = 5


class MassConservation(nn.Module):
    def forward(self, fine_pred: torch.Tensor, coarse_value: torch.Tensor) -> torch.Tensor:
        """fine_pred: (B, 1, H, W) - the network's raw prediction, must be non-negative already.
        coarse_value: (B, 1, H, W) - the ORIGINAL coarse rainfall, upsampled to fine resolution by simple
        repeat (nearest-neighbour, not bicubic - the true per-coarse-cell value each fine cell must average
        back to). Returns the rescaled fine field.
        """
        B, C, H, W = fine_pred.shape
        ch, cw = H // COARSEN_FACTOR, W // COARSEN_FACTOR

        fine_blocks = fine_pred.reshape(B, C, ch, COARSEN_FACTOR, cw, COARSEN_FACTOR)
        pred_block_mean = fine_blocks.mean(dim=(3, 5), keepdim=True)  # (B,C,ch,1,cw,1)

        coarse_blocks = coarse_value.reshape(B, C, ch, COARSEN_FACTOR, cw, COARSEN_FACTOR)
        target_block_mean = coarse_blocks.mean(dim=(3, 5), keepdim=True)

        # Rescale factor per coarse block; where the predicted block mean is ~0 (no rain predicted anywhere
        # in that block), rescaling would divide by ~0 and blow up - fall back to just using the target mean
        # directly (spread the coarse rain evenly) rather than an exploding ratio.
        eps = 1e-6
        safe_pred_mean = torch.where(pred_block_mean > eps, pred_block_mean, torch.full_like(pred_block_mean, eps))
        scale = target_block_mean / safe_pred_mean

        rescaled_blocks = fine_blocks * scale
        # Where the original predicted block mean was ~0 but the target says there SHOULD be rain, spreading
        # via a scale factor on all-zero predictions still gives all-zero (0 * anything = 0) - handle that
        # degenerate case by falling back to uniform redistribution of the coarse value.
        near_zero_pred = pred_block_mean <= eps
        uniform_fallback = target_block_mean.expand_as(fine_blocks)
        rescaled_blocks = torch.where(near_zero_pred.expand_as(fine_blocks), uniform_fallback, rescaled_blocks)

        return rescaled_blocks.reshape(B, C, H, W)
