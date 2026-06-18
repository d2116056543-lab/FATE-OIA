from __future__ import annotations

import math

import torch
from torch import nn

from .acpr_sparse_ops import entmax15_bisect


class ACPRSparseEvidenceCoAttention(nn.Module):
    """Action-specific sparse reason evidence residual.

    The residual gates are initialized to exactly zero, so inserting this
    module is functionally equivalent to ACPR-CalAlign until optimization
    updates the gates.
    """

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        num_heads: int = 4,
        max_residual_scale: float = 0.20,
        evidence_grad_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.max_residual_scale = float(max_residual_scale)
        self.evidence_grad_scale = float(evidence_grad_scale)

        self.pre_norm_action = nn.LayerNorm(dim)
        self.pre_norm_reason = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.null_reason_token = nn.Parameter(torch.empty(1, 1, dim))
        self.residual_gate_raw = nn.Parameter(torch.zeros(action_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)
        nn.init.normal_(self.null_reason_token, mean=0.0, std=0.02)
        with torch.no_grad():
            self.residual_gate_raw.zero_()

    def forward(self, action_nodes: torch.Tensor, reason_nodes: torch.Tensor) -> dict[str, torch.Tensor]:
        b, a, d = action_nodes.shape
        if a != self.action_dim or reason_nodes.shape[1] != self.reason_dim:
            raise ValueError("Unexpected action/reason node shape")
        action_norm = self.pre_norm_action(action_nodes)
        reason_scaled = reason_nodes.detach() + self.evidence_grad_scale * (reason_nodes - reason_nodes.detach())
        reason_norm = self.pre_norm_reason(reason_scaled)
        null = self.null_reason_token.to(reason_norm.dtype).expand(b, -1, -1)
        reason_with_null = torch.cat([reason_norm, null], dim=1)

        q = self.q_proj(action_norm).view(b, self.action_dim, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(reason_with_null).view(b, self.reason_dim + 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(reason_with_null).view(b, self.reason_dim + 1, self.num_heads, self.head_dim).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        attn_heads = entmax15_bisect(score, dim=-1)
        ctx = torch.matmul(attn_heads, v).transpose(1, 2).contiguous().view(b, self.action_dim, d)
        evidence_context = self.out_proj(ctx)
        residual_scale = torch.tanh(self.residual_gate_raw) * self.max_residual_scale
        action_nodes_seca = action_nodes + residual_scale.view(1, self.action_dim, 1) * evidence_context
        attn = attn_heads.mean(dim=1)
        no_null = attn[..., : self.reason_dim]
        null_attention = attn[..., self.reason_dim]
        support = (no_null > 1e-6).float().sum(-1)
        entropy = -(attn.clamp_min(1e-9).log() * attn).sum(-1)
        diversity = (no_null[:, :, None, :] - no_null[:, None, :, :]).abs().mean(dim=-1).mean(dim=(1, 2))
        return {
            "action_nodes_seca": action_nodes_seca,
            "action_reason_attention_heads": attn_heads,
            "action_reason_attention": attn,
            "action_reason_attention_no_null": no_null,
            "null_attention": null_attention,
            "residual_scale": residual_scale,
            "evidence_context": evidence_context,
            "evidence_context_norm": evidence_context.norm(dim=-1).mean(),
            "active_reason_count": support.mean(),
            "attention_entropy": entropy.mean(),
            "action_attention_diversity": diversity.mean(),
        }
