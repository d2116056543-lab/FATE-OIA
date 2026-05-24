from __future__ import annotations

import torch
from torch import nn


class ReasonToActionBottleneck(nn.Module):
    """Reason-probability bottleneck that contributes auxiliary action logits.

    It consumes sigmoid reason probabilities, not raw explanation text, so it
    can be ablated or detached without leaking test-time annotations.
    """

    def __init__(self, reason_dim: int = 21, action_dim: int = 4, hidden_dim: int = 128, detach_reason: bool = False) -> None:
        super().__init__()
        self.detach_reason = detach_reason
        self.net = nn.Sequential(nn.Linear(reason_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, reason_logits: torch.Tensor) -> torch.Tensor:
        reason_probs = torch.sigmoid(reason_logits)
        if self.detach_reason:
            reason_probs = reason_probs.detach()
        return self.net(reason_probs)