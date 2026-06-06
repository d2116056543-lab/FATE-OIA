from __future__ import annotations

import torch
from torch import nn


class MaskedReasonFromFactorTransformer(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, reason_cap: float = 0.25, num_heads: int = 4) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.reason_cap = float(reason_cap)
        self.reason_queries = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.reason_head = nn.Linear(dim, 1)
        self.gate = nn.Sequential(nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, selected_factor_tokens: torch.Tensor, actor_evidence_tokens: torch.Tensor, base_reason_logits: torch.Tensor, scene_state_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b = base_reason_logits.shape[0]
        selected_flat = selected_factor_tokens.reshape(b, -1, selected_factor_tokens.shape[-1])
        actor_flat = actor_evidence_tokens.reshape(b, -1, actor_evidence_tokens.shape[-1])
        context = torch.cat([selected_flat, actor_flat], dim=1)
        if scene_state_tokens is not None:
            context = torch.cat([context, scene_state_tokens], dim=1)
        q = self.reason_queries.unsqueeze(0).expand(b, -1, -1)
        out, attn = self.attn(q, context, context, need_weights=True, average_attn_weights=True)
        reason_factor = self.reason_head(out).squeeze(-1)
        support = attn[:, :, : selected_flat.shape[1]].mean(-1, keepdim=True)
        g_reason = self.gate(torch.cat([out, support.expand(-1, -1, 1)], dim=-1)).squeeze(-1)
        delta = torch.clamp(reason_factor - base_reason_logits, -self.reason_cap, self.reason_cap)
        final = base_reason_logits + g_reason * delta
        k = selected_factor_tokens.shape[2]
        reason_to_factor = attn[:, :, : self.action_dim * k].reshape(b, self.reason_dim, self.action_dim, k)
        return {
            "reason_factor_logits": reason_factor,
            "final_reason_logits": final,
            "reason_gate": g_reason,
            "reason_delta": delta,
            "reason_to_factor_attention": reason_to_factor,
            "factor_reason_support": support.squeeze(-1),
        }
