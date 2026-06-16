from __future__ import annotations

import torch
from torch import nn


class ACPRActionPredicateDelta(nn.Module):
    """Bounded predicate-to-action micro correction.

    This module only produces an action delta. It never reads or writes reason
    logits, and its output is clamped by max_delta so it cannot replace the
    fallback ACPR-CalAlign action branch.
    """

    def __init__(
        self,
        dim: int = 384,
        num_predicates: int = 32,
        action_dim: int = 4,
        hidden_dim: int = 192,
        max_delta: float = 0.05,
        detach_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_delta = float(max_delta)
        self.detach_inputs = bool(detach_inputs)
        self.norm_action = nn.LayerNorm(dim)
        self.norm_pred = nn.LayerNorm(num_predicates)
        self.mlp = nn.Sequential(
            nn.Linear(dim + int(num_predicates), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(int(hidden_dim), 1),
        )
        # V1.3 failed because a zero-initialized predicate branch produced an
        # exactly-zero candidate and never beat the fallback during gate search.
        # Keep the initial delta tiny, but non-zero, so candidate-probe training
        # has a real gradient path before any gate is opened.
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, action_nodes: torch.Tensor, predicate_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.detach_inputs:
            action_nodes = action_nodes.detach()
            predicate_probs = predicate_probs.detach()
        b, a, _ = action_nodes.shape
        pred = predicate_probs.unsqueeze(1).expand(b, a, predicate_probs.shape[-1])
        x = torch.cat([self.norm_action(action_nodes), self.norm_pred(pred)], dim=-1)
        raw = self.mlp(x).squeeze(-1)
        delta = torch.tanh(raw) * self.max_delta
        return {
            "predicate_action_delta_raw": raw,
            "predicate_action_delta": delta,
            "predicate_action_delta_abs_mean": delta.abs().mean(),
            "predicate_action_delta_max_abs": delta.abs().max(),
            "predicate_action_delta_per_action_mean": delta.mean(0),
        }
