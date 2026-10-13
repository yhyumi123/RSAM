"""Instance normalization for FP32 feature planes at rounding precision."""

import torch
from torch import nn


class StableInstanceNorm2d(nn.InstanceNorm2d):
    """Zero rounding residuals in constant FP32 feature planes."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__(
            num_features, eps=eps, affine=False, track_running_stats=False
        )

    def forward(self, value):
        if value.dtype != torch.float32:
            raise ValueError("Stable instance normalization requires FP32 tensors")
        result = super().forward(value)
        with torch.no_grad():
            high = value.amax((-2, -1), keepdim=True)
            low = value.amin((-2, -1), keepdim=True)
            magnitude = torch.maximum(high.abs(), low.abs()).clamp_min(1)
            constant = high - low <= 16 * torch.finfo(torch.float32).eps * magnitude
        return torch.where(constant, result - result.detach(), result)


def instance_norm(channels, stable=False):
    return StableInstanceNorm2d(channels) if stable else nn.InstanceNorm2d(channels)
