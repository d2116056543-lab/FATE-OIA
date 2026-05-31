from __future__ import annotations

import torch
from torch import nn


class EvidenceConditionedLabelCorr(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, heads: int = 4, rezero_init: float = 0.01, residual_max: float = 0.08, intervention_dropout: float = 0.20) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(6, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))
        self.dropout = nn.Dropout(intervention_dropout)
        self.rezero = nn.Parameter(torch.tensor(float(rezero_init)))
        self.residual_max = float(residual_max)

    def forward(self, reason_tokens: torch.Tensor, reliability: torch.Tensor, source_mass_by_reason: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if source_mass_by_reason is None:
            source_mass_by_reason = reason_tokens.new_zeros(reason_tokens.shape[0], reason_tokens.shape[1], 5)
        gate = torch.sigmoid(self.gate(torch.cat([reliability.unsqueeze(-1).to(reason_tokens.dtype), source_mass_by_reason.to(reason_tokens.dtype)], dim=-1)))
        attn_out, attn = self.attn(reason_tokens, reason_tokens, reason_tokens, need_weights=True)
        scale = torch.clamp(self.rezero, 0.0, self.residual_max)
        return {"reason_tokens": reason_tokens + scale * gate * self.dropout(attn_out), "label_corr_gate": gate.squeeze(-1), "label_corr_attention": attn, "label_corr_scale": scale}
