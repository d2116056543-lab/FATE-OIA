from __future__ import annotations

import torch
from torch import nn


EXPERT_TYPES = ["object", "lane", "drivable", "traffic_control", "global_context"]


class ExpertRouter(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, num_experts: int = 5, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.scorer = nn.Sequential(nn.Linear(dim + action_dim + 1, dim), nn.GELU(), nn.Linear(dim, num_experts))

    def forward(self, reason_tokens: torch.Tensor, base_action: torch.Tensor, base_reason: torch.Tensor, active_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        b, r, d = reason_tokens.shape
        action = base_action.unsqueeze(1).expand(-1, r, -1)
        rs = base_reason.unsqueeze(-1)
        logits = self.scorer(torch.cat([reason_tokens, action, rs], dim=-1))
        logits = logits.masked_fill(~active_mask.unsqueeze(-1), -1e9)
        top = torch.topk(logits, k=min(self.top_k, self.num_experts), dim=-1).indices
        route = torch.zeros_like(logits, dtype=torch.bool)
        route.scatter_(-1, top, True)
        route = route & active_mask.unsqueeze(-1)
        probs = torch.softmax(logits, dim=-1).masked_fill(~route, 0.0)
        denom = probs.sum(-1, keepdim=True).clamp_min(1e-6)
        probs = probs / denom
        return {"expert_logits": logits, "expert_route_mask": route, "expert_route_probs": probs, "top_expert_indices": top}
