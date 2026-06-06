from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class CAGEReasonReliability(nn.Module):
    """Per-reason reliability gate for noisy and long-tail BDD-OIA reasons."""

    def __init__(self, reason_dim: int = 21, hidden_dim: int = 32):
        super().__init__()
        self.reason_dim = reason_dim
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        reason_logits: torch.Tensor,
        evidence_confidence: torch.Tensor,
        selected_drop: torch.Tensor,
        label_frequency: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if reason_logits.shape[-1] != self.reason_dim:
            raise ValueError(f"expected {self.reason_dim} reason logits")
        bsz = reason_logits.shape[0]
        freq = label_frequency.to(reason_logits.device, reason_logits.dtype).view(1, self.reason_dim).expand(bsz, -1)
        features = torch.stack(
            [reason_logits.sigmoid(), evidence_confidence, selected_drop, freq],
            dim=-1,
        )
        reliability = torch.sigmoid(self.net(features).squeeze(-1))
        return {"reason_reliability": reliability, "features": features}
