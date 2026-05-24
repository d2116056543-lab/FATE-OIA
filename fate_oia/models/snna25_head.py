from __future__ import annotations

import torch
from torch import nn


class SNNA25Head(nn.Module):
    """Linear multi-task head for frozen SNNA/DINO features.

    The head predicts BDD-OIA action labels and 21 reason labels as independent
    sigmoid multi-label outputs. It intentionally does not use softmax.
    """

    def __init__(self, in_dim: int, action_dim: int = 4, reason_dim: int = 21, dropout: float = 0.0) -> None:
        super().__init__()
        if action_dim not in (4, 5):
            raise ValueError("action_dim must be 4 or 5")
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.action_head = nn.Linear(in_dim, action_dim)
        self.reason_head = nn.Linear(in_dim, reason_dim)
        self.reset_parameters()

    @property
    def num_labels(self) -> int:
        return self.action_dim + self.reason_dim

    def reset_parameters(self) -> None:
        for head in (self.action_head, self.reason_head):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        features = features.view(features.shape[0], -1)
        features = self.dropout(features)
        action_logits = self.action_head(features)
        reason_logits = self.reason_head(features)
        return {
            "action_logits": action_logits,
            "reason_logits": reason_logits,
            "logits": torch.cat([action_logits, reason_logits], dim=1),
        }