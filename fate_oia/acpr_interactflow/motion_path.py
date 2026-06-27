from __future__ import annotations

import torch
from torch import nn


class MotionPathEncoder(nn.Module):
    def __init__(self, dim: int = 384, hidden_dim: int = 256) -> None:
        super().__init__()
        self.gru = nn.GRU(dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, dim)
        self.speed_head = nn.Linear(dim, 3)

    def forward(self, anchor_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        seq, _ = self.gru(anchor_tokens)
        token = self.proj(seq[:, -1])
        return {
            "motion_token": token,
            "motion_sequence": seq,
            "motion_logits": self.speed_head(token),
        }

