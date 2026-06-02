from __future__ import annotations

import torch
from torch import nn


class ActionSetConsistencyHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, action_set_cap: float = 0.06) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_set_cap = action_set_cap
        prototypes = torch.tensor(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("action_set_prototypes", prototypes)
        self.net = nn.Sequential(nn.Linear(dim + action_dim * 2, dim), nn.GELU(), nn.Linear(dim, prototypes.shape[0]))

    def forward(self, base_action_logits: torch.Tensor, action_evidence_context: torch.Tensor, action_uncertainty: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = action_evidence_context.mean(1)
        x = torch.cat([pooled, base_action_logits, action_uncertainty], dim=-1)
        set_logits = self.net(x)
        proto_prob = torch.softmax(set_logits, dim=-1)
        action_prob = (proto_prob @ self.action_set_prototypes.to(proto_prob.device, proto_prob.dtype)).clamp(1e-4, 1.0 - 1e-4)
        action_set_logits = torch.logit(action_prob)
        delta = self.action_set_cap * torch.tanh((action_set_logits - base_action_logits) / 4.0)
        return {
            "action_set_logits": base_action_logits + delta,
            "action_set_delta": delta,
            "action_set_probs": proto_prob,
            "selected_action_set_id": proto_prob.argmax(-1),
        }
