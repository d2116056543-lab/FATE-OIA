from __future__ import annotations

import torch
from torch import nn


class ActionSetHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, num_prototypes: int = 8) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_prototypes = int(num_prototypes)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        self.score = nn.Linear(dim, num_prototypes)
        self.prototype_to_action = nn.Linear(num_prototypes, action_dim)

    def forward(self, action_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        summary = action_tokens.mean(dim=1)
        prototype_scores = self.score(summary)
        action_set_logits = self.prototype_to_action(prototype_scores)
        return {"action_set_logits": action_set_logits, "action_prototype_scores": prototype_scores}
