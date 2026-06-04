from __future__ import annotations

import torch
from torch import nn


class ActionSetHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, num_prototypes: int = 8) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        priors = torch.tensor(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        if action_dim != 4:
            priors = priors[:, :action_dim]
        self.num_prototypes = int(min(num_prototypes, priors.shape[0]))
        self.register_buffer("prototype_vectors", priors[: self.num_prototypes].clone())
        self.prototype_residual = nn.Parameter(torch.zeros_like(self.prototype_vectors))
        self.score = nn.Linear(dim, self.num_prototypes)
        self.residual = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, action_dim))

    def forward(self, action_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        summary = action_tokens.mean(dim=1)
        prototype_scores = self.score(summary)
        usage = torch.softmax(prototype_scores, dim=-1)
        proto = self.prototype_vectors + self.prototype_residual
        prior_logits = torch.matmul(prototype_scores, proto)
        action_set_logits = prior_logits + 0.05 * self.residual(summary)
        return {
            "action_set_logits": action_set_logits,
            "action_prototype_scores": prototype_scores,
            "action_prototype_usage": usage,
            "action_prototype_usage_mean": usage.mean(dim=0),
        }
