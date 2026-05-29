from __future__ import annotations

import torch
from torch import nn


class EvidenceAuxiliary(nn.Module):
    """Optional true-evidence auxiliary branch.

    This module never fabricates scene-token evidence. If no real evidence tokens
    are supplied, it returns None and a zero evidence count.
    """

    def __init__(self, dim: int = 384, reason_dim: int = 21) -> None:
        super().__init__()
        self.reason_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, reason_dim))

    def forward(self, evidence_tokens: torch.Tensor | None) -> tuple[torch.Tensor | None, dict[str, float | int]]:
        if evidence_tokens is None or evidence_tokens.numel() == 0:
            return None, {"evidence_count": 0, "evidence_available": 0}
        pooled = evidence_tokens.mean(1)
        return self.reason_head(pooled), {"evidence_count": int(evidence_tokens.shape[1]), "evidence_available": 1}
