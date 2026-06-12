from __future__ import annotations

import itertools

import torch
from torch import nn


ACTION_NAMES = ["forward", "stop", "left", "right"]
PAIR_INDEX = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def build_subset_membership(action_dim: int = 4) -> torch.Tensor:
    rows = []
    for subset_id in range(2 ** action_dim):
        rows.append([(subset_id >> i) & 1 for i in range(action_dim)])
    return torch.tensor(rows, dtype=torch.float32)


def action_targets_to_subset_ids(action_targets: torch.Tensor) -> torch.Tensor:
    bits = (action_targets.float() > 0.5).long()
    weights = torch.tensor([1, 2, 4, 8], device=bits.device, dtype=torch.long)
    return (bits * weights.view(1, -1)).sum(-1)


class CastActionSetEnergy(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.atomic_head = nn.Linear(dim, 1)
        self.pair_head = nn.Linear(dim * 4, 6)
        self.cardinality_bias = nn.Parameter(torch.zeros(action_dim + 1))
        self.set_context_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2 ** action_dim))
        self.register_buffer("subset_membership", build_subset_membership(action_dim), persistent=False)
        pair_mask = torch.zeros(2 ** action_dim, len(PAIR_INDEX))
        subset = self.subset_membership
        for s in range(2 ** action_dim):
            for j, (a, b) in enumerate(PAIR_INDEX):
                pair_mask[s, j] = subset[s, a] * subset[s, b]
        self.register_buffer("subset_pair_membership", pair_mask, persistent=False)

    def forward(self, action_nodes: torch.Tensor, graph_context: torch.Tensor, subset_context: torch.Tensor | None = None) -> dict:
        b = action_nodes.shape[0]
        atomic_logits = self.atomic_head(action_nodes).squeeze(-1)
        action_summary = action_nodes.mean(1)
        pair_feat = torch.cat(
            [action_summary, graph_context, action_summary * graph_context, torch.abs(action_summary - graph_context)],
            dim=-1,
        )
        pair_logits = self.pair_head(pair_feat)
        subset = self.subset_membership.to(action_nodes.device, action_nodes.dtype)
        pair_subset = self.subset_pair_membership.to(action_nodes.device, action_nodes.dtype)
        card = subset.sum(-1).long()
        graph_set_context = self.set_context_head(graph_context)
        if subset_context is not None:
            graph_set_context = graph_set_context + subset_context.mean(-1)
        action_set_logits = (
            subset @ atomic_logits.t()
        ).t() + (pair_subset @ pair_logits.t()).t() + self.cardinality_bias[card].view(1, -1) + graph_set_context
        action_set_probs = torch.softmax(action_set_logits, dim=-1)
        action_marginal_probs = action_set_probs @ subset
        action_logits = torch.logit(action_marginal_probs.clamp(1e-5, 1 - 1e-5))
        card_probs = torch.zeros(b, self.action_dim + 1, device=action_nodes.device, dtype=action_nodes.dtype)
        for c in range(self.action_dim + 1):
            card_probs[:, c] = action_set_probs[:, card == c].sum(-1)
        return {
            "atomic_logits": atomic_logits,
            "pair_logits": pair_logits,
            "cardinality_logits": torch.logit(card_probs.clamp(1e-5, 1 - 1e-5)),
            "action_set_logits": action_set_logits,
            "action_set_probs": action_set_probs,
            "action_marginal_probs": action_marginal_probs,
            "action_logits": action_logits,
            "graph_set_context": graph_set_context,
        }
