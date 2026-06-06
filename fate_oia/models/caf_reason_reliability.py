from __future__ import annotations

import torch
from torch import nn


class ReasonReliabilityGate(nn.Module):
    def __init__(self, reason_dim: int = 21) -> None:
        super().__init__()
        self.reason_dim = int(reason_dim)
        self.freq_bias = nn.Parameter(torch.zeros(reason_dim))

    def forward(
        self,
        reason_logits: torch.Tensor,
        factor_support: torch.Tensor,
        base_reason_logits: torch.Tensor | None = None,
        scene_state_alignment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uncertainty = 1.0 - (torch.sigmoid(reason_logits) - 0.5).abs() * 2.0
        support = factor_support.float()
        if scene_state_alignment is not None:
            support = support + scene_state_alignment.float()
        # base_reason_logits is accepted for diagnostics/compatibility but is not used as support.
        return torch.sigmoid(support + uncertainty + self.freq_bias)
