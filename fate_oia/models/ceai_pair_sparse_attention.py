from __future__ import annotations

import torch
from torch import nn


DEFAULT_REASON_GROUPS = {
    "traffic_control": [0, 3, 4, 13],
    "obstacle": [5, 6, 7, 8, 10, 14, 16],
    "lane_left": [9, 10, 11, 12, 13, 14],
    "lane_right": [15, 16, 17],
    "drivable": [1, 2, 18, 19, 20],
    "global_tail": [5, 6, 9, 10, 11, 12, 13, 14],
}


def default_reason_to_group(reason_dim: int = 21) -> list[int]:
    mapping = [5 for _ in range(reason_dim)]
    for g, ids in enumerate(DEFAULT_REASON_GROUPS.values()):
        for idx in ids:
            if 0 <= idx < reason_dim:
                mapping[idx] = g
    return mapping


def group_reason_tokens(reason_tokens: torch.Tensor, reason_to_group: list[int] | None = None, group_count: int = 6) -> torch.Tensor:
    reason_to_group = reason_to_group or default_reason_to_group(reason_tokens.shape[1])
    groups = []
    for g in range(group_count):
        idx = [i for i, m in enumerate(reason_to_group) if m == g and i < reason_tokens.shape[1]]
        if idx:
            groups.append(reason_tokens[:, idx].mean(dim=1))
        else:
            groups.append(reason_tokens.mean(dim=1))
    return torch.stack(groups, dim=1)


class TaskGuidedPairSparseAttention(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_group_count: int = 6, topk: int = 24, temperature: float = 0.7, heads: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_group_count = reason_group_count
        self.topk = topk
        self.temperature = temperature
        self.action_proj = nn.Linear(dim, dim)
        self.reason_proj = nn.Linear(dim, dim)
        self.scene_proj = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, action_tokens: torch.Tensor, reason_group_tokens: torch.Tensor, scene_state_tokens: torch.Tensor, visual_tokens: torch.Tensor) -> dict[str, torch.Tensor | dict[str, float]]:
        b, n, d = visual_tokens.shape
        scene = scene_state_tokens.mean(dim=1)
        q = (
            self.action_proj(action_tokens).unsqueeze(2)
            + self.reason_proj(reason_group_tokens).unsqueeze(1)
            + self.scene_proj(scene).view(b, 1, 1, d)
        )
        keys = self.key(visual_tokens)
        values = self.value(visual_tokens)
        scores = torch.einsum("bagd,bnd->bagn", q, keys) / (d ** 0.5)
        k = min(max(1, self.topk), n)
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        weights = torch.softmax(top_scores / max(self.temperature, 1e-4), dim=-1)
        expanded_values = values[:, None, None].expand(-1, self.action_dim, self.reason_group_count, -1, -1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d)
        top_values = torch.gather(expanded_values, 3, gather_idx)
        context = (weights.unsqueeze(-1) * top_values).sum(dim=-2)
        context = self.out(context)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        concentration = weights.max(dim=-1).values
        stats = {
            "pair_attention_entropy": float(entropy.detach().mean().cpu()),
            "pair_attention_concentration": float(concentration.detach().mean().cpu()),
            "pair_attention_topk": int(k),
        }
        return {
            "pair_group_context": context,
            "attention_indices": top_idx,
            "attention_weights": weights,
            "attention_entropy": entropy,
            "attention_concentration": concentration,
            "stats": stats,
        }
