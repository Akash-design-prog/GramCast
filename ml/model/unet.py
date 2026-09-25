"""Residual U-Net, per the project guide section 5: input is bicubic-upsampled coarse rainfall stacked with
terrain/land-cover channels, the network predicts a CORRECTION added on top of the bicubic rainfall channel
(input channel 0), not the whole field from scratch - this is easier to learn since the bicubic field is
already a reasonable starting point.

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

        self.correction_head = nn.Conv2d(base_channels, 1, 1)  # predicts the correction, 1 output channel
        self.mass_conservation = MassConservation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, 35, 50), channel 0 is the bicubic rainfall (already normalized for the encoder, but the
        residual/mass-conservation steps need the RAW unnormalized value - see predict() wrapper below, which
        handles denormalization; this forward() operates purely on already-prepared tensors for training)."""
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

        correction_padded = self.correction_head(d1)
        correction = correction_padded[:, :, :H, :W]  # crop back to original 35x50
        return correction

    def predict_rainfall(self, x: torch.Tensor, bicubic_raw: torch.Tensor, coarse_nn_raw: torch.Tensor) -> torch.Tensor:
        """Full inference path: correction -> add to bicubic -> clip non-negative -> mass-conserve.
        bicubic_raw: (B,1,H,W) UNnormalized bicubic rainfall (channel 0 of x, denormalized).
        coarse_nn_raw: (B,1,H,W) UNnormalized nearest-neighbour coarse rainfall (the true per-block target
        the mass-conservation step must preserve - NOT the bicubic value, which doesn't conserve mass itself).
        """
        correction = self.forward(x)
        raw_pred = bicubic_raw + correction
        raw_pred = torch.clamp(raw_pred, min=0.0)  # rain can't be negative - same discipline as the bicubic fix
        conserved = self.mass_conservation(raw_pred, coarse_nn_raw)
        return conserved
