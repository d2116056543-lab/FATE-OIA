from __future__ import annotations

import torch
from torch import nn


class StrongLabelDecoder(nn.Module):
    """Query2Label-style TransformerDecoder over label queries and DINO tokens."""

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
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=max(1, int(self_layers)))
        self.out_norm = nn.LayerNorm(dim)
        self.label_classifier = nn.Linear(dim, 1)

    def forward(self, label_queries: torch.Tensor, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if label_queries.ndim != 3 or tokens.ndim != 3:
            raise ValueError("label_queries and tokens must be [B,L,D] and [B,N,D]")
        memory = self.token_norm(tokens)
        tgt = self.query_norm(label_queries)
        label_tokens = self.decoder(tgt=tgt, memory=memory)
        label_tokens = self.out_norm(label_tokens)
        logits = self.label_classifier(label_tokens).squeeze(-1)
        # nn.TransformerDecoder does not expose cross-attention weights, so expose a
        # deterministic label-token/video-token compatibility map for diagnostics.
        attention = torch.softmax(torch.einsum("bld,bnd->bln", label_tokens, memory) / (self.dim ** 0.5), dim=-1)
        return {
            "logits": logits,
            "action_logits": logits[:, : self.action_dim],
            "reason_logits": logits[:, self.action_dim :],
            "label_tokens": label_tokens,
            "attention": attention,
        }
