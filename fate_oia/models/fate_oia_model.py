from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.label_query_head import LabelQueryHead


class FATEOIAFeatureModel(nn.Module):
    """Feature-level FATE-OIA head for SNNA/ViT token features.

    It expects precomputed or backbone-produced tokens [B,N,D]. This keeps the module
    compatible with SNNA checkpoints that are still being trained.
    """

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, use_label_query: bool = True) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.use_label_query = use_label_query
        if use_label_query:
            self.label_head = LabelQueryHead(dim, action_dim + reason_dim)
        else:
            self.pool = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
            self.action_head = nn.Linear(dim, action_dim)
            self.reason_head = nn.Linear(dim, reason_dim)
        self.reason_to_action = nn.Sequential(nn.Linear(reason_dim, dim), nn.GELU(), nn.Linear(dim, action_dim))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.use_label_query:
            out = self.label_head(tokens)
            logits = out["logits"]
            action_logits = logits[:, : self.action_dim]
            reason_logits = logits[:, self.action_dim :]
            r2a = self.reason_to_action(torch.sigmoid(reason_logits))
            return {**out, "action_logits": action_logits, "reason_logits": reason_logits, "reason_to_action_logits": r2a}
        pooled = self.pool(tokens.mean(1))
        reason_logits = self.reason_head(pooled)
        return {
            "action_logits": self.action_head(pooled),
            "reason_logits": reason_logits,
            "reason_to_action_logits": self.reason_to_action(torch.sigmoid(reason_logits)),
        }
