import torch
from torch import nn


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.privacy_cloud.silu_and_mul(x)
