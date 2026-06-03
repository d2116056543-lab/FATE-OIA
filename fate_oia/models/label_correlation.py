from __future__ import annotations

import torch
from torch import nn


class LabelCorrelationLayer(nn.Module):
    """Self-attention over label tokens for multi-label dependency modeling."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, ffn_ratio: float = 2.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        hidden = max(dim, int(dim * ffn_ratio))
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, label_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm_attn(label_tokens)
        attn_out, attn = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        label_tokens = label_tokens + attn_out
        label_tokens = label_tokens + self.ffn(self.norm_ffn(label_tokens))
        return label_tokens, attn


class LabelCorrelationBlock(nn.Module):
    """Query2Label-style label-token refinement without changing the visual backbone."""

    def __init__(self, dim: int, num_layers: int = 1, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList(
            [LabelCorrelationLayer(dim=dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, label_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        attentions: list[torch.Tensor] = []
        x = label_tokens
        for layer in self.layers:
            x, attn = layer(x)
            attentions.append(attn)
        x = self.norm(x)
        return {"label_tokens": x, "attention": torch.stack(attentions, dim=1)}
