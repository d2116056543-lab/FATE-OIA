from __future__ import annotations

import torch
from torch import nn


class StrongLabelDecoder(nn.Module):
    """Query2Label-style cross-attention plus label self-attention decoder."""

    def __init__(
        self,
        *,
        dim: int,
        action_dim: int = 4,
        reason_dim: int = 21,
        num_heads: int = 4,
        self_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_labels = self.action_dim + self.reason_dim
        self.query_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(0, int(self_layers)))
        self.out_norm = nn.LayerNorm(dim)
        self.label_classifier = nn.Linear(dim, 1)

    def forward(self, label_queries: torch.Tensor, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if label_queries.ndim != 3 or tokens.ndim != 3:
            raise ValueError("label_queries and tokens must be [B,L,D] and [B,N,D]")
        attended, attn = self.cross_attn(
            self.query_norm(label_queries),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=True,
            average_attn_weights=True,
        )
        label_tokens = label_queries + attended
        label_tokens = self.self_encoder(label_tokens)
        label_tokens = self.out_norm(label_tokens)
        logits = self.label_classifier(label_tokens).squeeze(-1)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return {
            "logits": logits,
            "action_logits": logits[:, : self.action_dim],
            "reason_logits": logits[:, self.action_dim :],
            "label_tokens": label_tokens,
            "attention": attn,
        }
