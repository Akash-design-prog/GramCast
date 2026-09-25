"""Residual U-Net, per the project guide section 5: input is bicubic-upsampled coarse rainfall stacked with
terrain/land-cover channels, the network predicts a CORRECTION added on top of the bicubic rainfall channel
(input channel 0), not the whole field from scratch - this is easier to learn since the bicubic field is
already a reasonable starting point.

Quantile output (P10/P50/P90), per the project guide section 5 and its own "uncertainty as a headline, not a
footnote" framing (section 8c) - NOT optional. The network outputs 3 channels: a P50 correction plus two
NON-NEGATIVE deltas (via softplus). P10 = P50 - delta_low, P90 = P50 + delta_high. This guarantees
P10 <= P50 <= P90 BY CONSTRUCTION, not by hoping a pinball loss eventually learns it - a raw 3-channel output
trained independently per quantile has no such guarantee and can cross (P10 > P90) on real data, which is a
known, well-documented failure mode of naive quantile regression.

Grid note: the training grid is 35x50, not evenly divisible by 2^depth for a 2-level U-Net (needs divisible by
4). Pads to 36x52 (reflect padding, not zero - zero-padding would introduce a fake "zero rain" border the
network could learn to exploit) before the encoder, crops back to 35x50 after the decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mass_conservation import MassConservation

PAD_H, PAD_W = 36, 52  # next multiple of 4 >= 35, 50


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualUNet(nn.Module):
    """2-level U-Net (downsample x4 at the bottleneck) - modest depth, appropriate for a 35x50 grid where a
    deeper network would downsample to near-nothing."""

    def __init__(self, in_channels: int, base_channels: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # 3 output channels: P50 correction, log-delta-low, log-delta-high (softplus'd to enforce non-negativity)
        self.correction_head = nn.Conv2d(base_channels, 3, 1)
        self.mass_conservation = MassConservation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, 35, 50), channel 0 is the bicubic rainfall (already normalized for the encoder, but the
        residual/mass-conservation steps need the RAW unnormalized value - see predict() wrapper below, which
        handles denormalization; this forward() operates purely on already-prepared tensors for training).
        Returns (B, 3, 35, 50): [p50_correction, delta_low_raw, delta_high_raw] - deltas are pre-softplus, not
        yet guaranteed non-negative (that happens in predict_quantiles, which is where softplus is applied)."""
        B, C, H, W = x.shape
        pad_h, pad_w = PAD_H - H, PAD_W - W
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        e1 = self.enc1(x_padded)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out_padded = self.correction_head(d1)
        return out_padded[:, :, :H, :W]  # crop back to original 35x50

    def predict_quantiles(self, x: torch.Tensor, bicubic_raw: torch.Tensor, coarse_nn_raw: torch.Tensor) -> dict:
        """Full inference path, quantile version.
        bicubic_raw: (B,1,H,W) UNnormalized bicubic rainfall (channel 0 of x, denormalized).
        coarse_nn_raw: (B,1,H,W) UNnormalized nearest-neighbour coarse rainfall - the mass-conservation target,
        applied to P50 only (the physical redistribution constraint applies to the central estimate; P10/P90
        represent uncertainty about that estimate, not independent physical scenarios each needing their own
        mass-conservation - constraining them too would collapse the uncertainty band to zero width at the
        coarse-block level, defeating the purpose of having one).
        Returns {"p10":..., "p50":..., "p90":...}, each (B,1,H,W), with p10<=p50<=p90 guaranteed by construction.
        """
        raw = self.forward(x)
        p50_correction = raw[:, 0:1, :, :]
        delta_low = F.softplus(raw[:, 1:2, :, :])
        delta_high = F.softplus(raw[:, 2:3, :, :])

        p50_raw = torch.clamp(bicubic_raw + p50_correction, min=0.0)
        p50 = self.mass_conservation(p50_raw, coarse_nn_raw)

        p10 = torch.clamp(p50 - delta_low, min=0.0)
        p90 = p50 + delta_high
        return {"p10": p10, "p50": p50, "p90": p90}

    def predict_rainfall(self, x: torch.Tensor, bicubic_raw: torch.Tensor, coarse_nn_raw: torch.Tensor) -> torch.Tensor:
        """Backward-compatible single-point prediction (P50 only) - used by code that only needs a point
        estimate (e.g. the existing metrics/comparison scripts built for the baselines)."""
        return self.predict_quantiles(x, bicubic_raw, coarse_nn_raw)["p50"]
