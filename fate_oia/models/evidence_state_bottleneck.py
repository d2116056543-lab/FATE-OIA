from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class EvidenceStateBottleneck(nn.Module):
    """Cross-attend a small set of evidence-state queries to patch/evidence tokens."""

    DEFAULT_STATE_NAMES = (
        "traffic_control_state",
        "obstacle_state",
        "vulnerable_agent_state",
        "lane_affordance_state",
        "drivable_space_state",
        "ego_path_conflict_state",
        "scene_context_state",
        "uncertainty_state",
    )

    def __init__(self, dim: int, num_state_queries: int = 8, num_heads: int = 4, dropout: float = 0.1, self_attn: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_state_queries = int(num_state_queries)
        self.state_queries = nn.Parameter(torch.randn(num_state_queries, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True) if self_attn else None
        self.self_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        evidence_tokens: torch.Tensor | None = None,
        evidence_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = patch_tokens.shape[0]
        if evidence_tokens is not None and evidence_tokens.numel() > 0:
            tokens = torch.cat([patch_tokens, evidence_tokens], dim=1)
            if evidence_mask is None:
                evidence_mask = torch.ones(evidence_tokens.shape[:2], dtype=torch.bool, device=patch_tokens.device)
            patch_mask = torch.ones(patch_tokens.shape[:2], dtype=torch.bool, device=patch_tokens.device)
            key_mask = ~torch.cat([patch_mask, evidence_mask.bool()], dim=1)
        else:
            tokens = patch_tokens
            key_mask = None
        q = self.state_queries.unsqueeze(0).expand(b, -1, -1)
        attended, weights = self.cross_attn(q, tokens, tokens, key_padding_mask=key_mask, need_weights=True, average_attn_weights=False)
        states = self.cross_norm(q + attended)
        if self.self_attn is not None:
            ss, _ = self.self_attn(states, states, states, need_weights=False)
            states = self.self_norm(states + ss)
        states = self.ffn_norm(states + self.ffn(states))
        # weights [B,H,K,T]
        probs = weights.float().clamp_min(1e-9)
        entropy = -(probs * probs.log()).sum(-1).mean(dim=(1, 2)) / math.log(max(tokens.shape[1], 2))
        return {"state_tokens": states, "state_attention": weights, "state_attention_entropy": entropy}
