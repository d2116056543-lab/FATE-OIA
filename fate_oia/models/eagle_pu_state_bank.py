from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .eagle_pu_sparse_ops import entmax15_bisect


class ObjectiveEnvironmentStateBank(nn.Module):
    state_groups = [
        "traffic_control", "front_object", "lateral_lane", "drivable_corridor", "road_geometry", "global_context",
        "traffic_red", "traffic_green", "stop_sign", "front_vehicle", "pedestrian", "rider",
        "left_lane", "right_lane", "left_blocked", "right_blocked", "solid_left", "solid_right",
        "left_turn", "right_turn", "parked_vehicle", "obstacle", "clear_road", "ego_motion",
    ]

    def __init__(self, dim: int = 384, num_layers: int = 3, num_states: int = 24) -> None:
        super().__init__()
        self.dim = dim
        self.num_states = num_states
        self.state_queries = nn.Parameter(torch.randn(num_states, dim) * 0.02)
        self.layer_router = nn.Parameter(torch.zeros(num_states, num_layers))
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.state_logit = nn.Linear(dim, 1)
        self.group_logit = nn.Linear(dim, 1)

    def forward(self, patch_tokens_by_layer: torch.Tensor, ego_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict[str, float]]:
        b, s, n, d = patch_tokens_by_layer.shape
        layer_weights = torch.softmax(self.layer_router, dim=-1)
        tokens = torch.einsum("gs,bsnd->bgnd", layer_weights, patch_tokens_by_layer)
        if ego_tokens is not None:
            tokens = tokens + ego_tokens.unsqueeze(1)
        q = self.state_queries.unsqueeze(0).expand(b, -1, -1)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)
        scores = torch.einsum("bgd,bgnd->bgn", q, k) / (d ** 0.5)
        attn = entmax15_bisect(scores, dim=-1)
        state_tokens = torch.einsum("bgn,bgnd->bgd", attn, v)
        state_logits = self.state_logit(state_tokens).squeeze(-1)
        state_group_logits = self.group_logit(state_tokens).squeeze(-1)
        entropy = (-(attn.clamp_min(1e-8).log() * attn).sum(-1)).mean()
        support = (attn > 1e-4).float().sum(-1).mean()
        stats = {"state_attention_entropy": float(entropy.detach().cpu()), "state_support_size": float(support.detach().cpu())}
        return {
            "state_tokens": state_tokens,
            "state_logits": state_logits,
            "state_attention": attn,
            "state_group_logits": state_group_logits,
            "state_layer_weights": layer_weights,
            "state_stats": stats,
        }
