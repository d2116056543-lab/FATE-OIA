from __future__ import annotations

import torch
from torch import nn


class NativeTextActionPredicateHead(nn.Module):
    def __init__(self, num_predicates: int, dim: int = 384, action_dim: int = 4, enabled_epoch: int = 6, cap_max: float = 0.05) -> None:
        super().__init__()
        self.enabled_epoch = int(enabled_epoch)
        self.cap_max = float(cap_max)
        self.action_queries = nn.Parameter(torch.randn(action_dim, dim) * 0.02)
        self.q_proj = nn.Linear(num_predicates + action_dim, action_dim)
        self.token_proj = nn.Linear(dim, dim)

    def forward(self, base_action_logits: torch.Tensor, q_pred: torch.Tensor, rho_pred: torch.Tensor, predicate_tokens: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor | dict]:
        if epoch < self.enabled_epoch:
            delta = torch.zeros_like(base_action_logits)
        else:
            token_score = torch.einsum("ad,bpd->bap", self.action_queries, self.token_proj(predicate_tokens)) / (predicate_tokens.shape[-1] ** 0.5)
            support = (torch.softmax(token_score, dim=-1) * q_pred.unsqueeze(1) * rho_pred.unsqueeze(1)).sum(-1)
            delta = torch.tanh(self.q_proj(torch.cat([q_pred * rho_pred, support], dim=-1))) * self.cap_max
        return {"action_predicate_delta": delta, "action_predicate_stats": {"action_predicate_delta_abs_mean": float(delta.abs().mean().detach().cpu()), "action_predicate_enabled": bool(epoch >= self.enabled_epoch)}}
