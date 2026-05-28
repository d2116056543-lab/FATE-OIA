from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.adaptformer_lite import AdaptFormerLite


class AdaptFormerDINOBlockWrapper(nn.Module):
    """Wrap a frozen ViT block with a trainable AdaptFormer residual adapter."""

    def __init__(self, block: nn.Module, dim: int = 384, bottleneck_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = block
        for param in self.block.parameters():
            param.requires_grad = False
        self.adapter = AdaptFormerLite(dim, bottleneck_dim=bottleneck_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.block(x))
