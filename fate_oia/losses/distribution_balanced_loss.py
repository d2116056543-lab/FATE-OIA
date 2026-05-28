from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DistributionBalancedBCE(nn.Module):
    """Lightweight optional pos-weight BCE placeholder for ScoreV2 ablations."""

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if pos_weight is None:
            self.register_buffer("pos_weight", torch.empty(0), persistent=True)
        else:
            self.register_buffer("pos_weight", pos_weight.float(), persistent=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight if self.pos_weight.numel() else None
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)
