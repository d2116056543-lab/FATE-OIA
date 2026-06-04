from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SceneStatePrototypeTransformer(nn.Module):
    """Learn image-only scene-state and implicit prototype tokens from visual tokens."""

    def __init__(
        self,
        dim: int = 384,
        scene_proto_count: int = 12,
        implicit_proto_count: int = 12,
        num_scene_states: int = 11,
        num_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.scene_proto_count = scene_proto_count
        self.implicit_proto_count = implicit_proto_count
        self.num_scene_states = num_scene_states
        self.scene_queries = nn.Parameter(torch.randn(scene_proto_count, dim) * 0.02)
        self.implicit_queries = nn.Parameter(torch.randn(implicit_proto_count, dim) * 0.02)
        self.visual_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.post = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.scene_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_scene_states))

    def _attend(self, queries: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = tokens.shape[0]
        q = queries.unsqueeze(0).expand(b, -1, -1)
        attended, weights = self.cross_attn(self.query_norm(q), self.visual_norm(tokens), self.visual_norm(tokens), need_weights=True, average_attn_weights=False)
        attended = attended + self.post(attended)
        return attended, weights

    def forward(self, visual_tokens: torch.Tensor) -> dict[str, torch.Tensor | dict[str, float]]:
        scene_tokens, scene_attn = self._attend(self.scene_queries, visual_tokens)
        implicit_tokens, implicit_attn = self._attend(self.implicit_queries, visual_tokens)
        scene_state_logits = self.scene_head(scene_tokens.mean(dim=1))
        stats = {
            "scene_attention_entropy": float(_attention_entropy(scene_attn).detach().mean().cpu()),
            "implicit_attention_entropy": float(_attention_entropy(implicit_attn).detach().mean().cpu()),
        }
        return {
            "scene_state_tokens": scene_tokens,
            "implicit_prototypes": implicit_tokens,
            "scene_state_logits": scene_state_logits,
            "scene_attention": scene_attn,
            "implicit_attention": implicit_attn,
            "scene_attention_stats": stats,
        }


def _attention_entropy(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = attn.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def masked_scene_state_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    masked = loss * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return masked.sum() / denom


def scene_state_stats(logits: torch.Tensor, target: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "scene_state_logit_mean": float(logits.detach().mean().cpu()),
        "scene_state_logit_std": float(logits.detach().std(unbiased=False).cpu()),
    }
    if target is not None and mask is not None:
        stats["scene_state_valid_count"] = int(mask.detach().sum().cpu())
        stats["scene_state_positive_count"] = int((target.detach() * mask.detach()).sum().cpu())
    return stats
