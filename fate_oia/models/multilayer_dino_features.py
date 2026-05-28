from __future__ import annotations

import torch
from torch import nn


class MultiLayerDINOFeatureFusion(nn.Module):
    """Learned weighted fusion for the last N DINO token layers."""

    def __init__(self, dim: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, layers: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        if len(layers) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {len(layers)}")
        shapes = [tuple(x.shape) for x in layers]
        if len(set(shapes)) != 1:
            raise ValueError(f"All DINO layers must have the same shape, got {shapes}")
        if layers[0].shape[-1] != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {layers[0].shape[-1]}")
        weights = torch.softmax(self.layer_logits, dim=0).to(device=layers[0].device, dtype=layers[0].dtype)
        stacked = torch.stack([x for x in layers], dim=0)
        fused = (weights.view(-1, 1, 1, 1) * stacked).sum(dim=0)
        fused = self.dropout(self.norm(fused))
        return {"tokens": fused, "layer_weights": weights.detach().cpu()}
