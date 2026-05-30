from __future__ import annotations

import torch
from torch import nn


class CausalLabelGrouping(nn.Module):
    """Small label-token self-attention block with near-identity initialization."""

    def __init__(self, dim: int = 384, num_heads: int = 4, residual_scale_init: float = 0.015, residual_scale_max: float = 0.08) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.residual_scale_max = float(residual_scale_max)

    def forward(self, reason_tokens: torch.Tensor, base_reason_logits: torch.Tensor | None = None, evidence_quality: torch.Tensor | None = None, training: bool = False) -> torch.Tensor:
        updated, _ = self.attn(reason_tokens, reason_tokens, reason_tokens, need_weights=False)
        updated = self.ffn(self.norm(updated))
        scale = torch.clamp(self.logit_scale, 0.0, self.residual_scale_max)
        if evidence_quality is not None:
            gate = evidence_quality.to(reason_tokens.device).to(reason_tokens.dtype).unsqueeze(-1)
            if gate.shape[1] != reason_tokens.shape[1]:
                gate = gate.mean(1, keepdim=True).expand(-1, reason_tokens.shape[1], -1)
        else:
            gate = 1.0
        return reason_tokens + scale * updated * gate

