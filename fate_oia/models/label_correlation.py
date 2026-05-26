from __future__ import annotations

import torch
from torch import nn


class LabelCorrelationBlock(nn.Module):
    """Self-attention over label tokens for action/reason correlation modeling."""

    def __init__(
        self,
        dim: int,
        num_labels: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        bias_mode: str = "none",
    ) -> None:
        super().__init__()
        if bias_mode != "none":
            raise ValueError("Only bias_mode='none' is implemented for launch-safe runs.")
        self.num_labels = int(num_labels)
        self.bias_mode = bias_mode
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(dim)

    def forward(self, label_tokens: torch.Tensor) -> torch.Tensor:
        if label_tokens.ndim != 3:
            raise ValueError("label_tokens must be [B,L,D]")
        if label_tokens.shape[1] != self.num_labels:
            raise ValueError(f"Expected {self.num_labels} label tokens, got {label_tokens.shape[1]}")
        return self.norm(self.encoder(label_tokens))
