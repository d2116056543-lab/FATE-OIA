from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def reason_delta_cap(epoch: int, max_cap: float = 0.15) -> float:
    if epoch <= 2:
        return 0.0
    if epoch <= 5:
        return min(max_cap, 0.05)
    if epoch <= 10:
        return min(max_cap, 0.10)
    return min(max_cap, 0.15)


class TFCReasonHead(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, max_delta: float = 0.15) -> None:
        super().__init__()
        self.reason_dim = int(reason_dim)
        self.max_delta = float(max_delta)
        self.reason_queries = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        self.visual_proj = nn.Linear(dim, dim)
        self.visual_head = nn.Linear(dim, 1)
        self.delta_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.gate = nn.Sequential(nn.Linear(3, 8), nn.GELU(), nn.Linear(8, 1))

    def visual_logits_from_patch(self, patch_reason: torch.Tensor) -> torch.Tensor:
        patch = patch_reason.mean(1)
        q = F.normalize(self.reason_queries, dim=-1)
        k = F.normalize(self.visual_proj(patch), dim=-1)
        attn = torch.softmax(torch.einsum("rd,bnd->brn", q, k), dim=-1)
        features = torch.einsum("brn,bnd->brd", attn, patch)
        return self.visual_head(features).squeeze(-1)

    def forward(
        self,
        patch_reason: torch.Tensor,
        factor_features_reason: torch.Tensor,
        credit_reason_norm: torch.Tensor,
        credit_confidence_reason: torch.Tensor,
        pu_state: dict,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        reason_visual_logits = self.visual_logits_from_patch(patch_reason)
        factor_target = torch.einsum("bfr,bfd->brd", credit_reason_norm, factor_features_reason)
        cap = reason_visual_logits.new_tensor(reason_delta_cap(epoch, self.max_delta))
        if float(cap.detach().cpu()) <= 0.0:
            delta = torch.zeros_like(reason_visual_logits)
            gate = torch.zeros_like(reason_visual_logits)
        else:
            support = pu_state.get("support_credit", torch.zeros_like(reason_visual_logits)).detach()
            contra = pu_state.get("contra_credit", torch.zeros_like(reason_visual_logits)).detach()
            gate_in = torch.stack([credit_confidence_reason.detach(), support, contra], dim=-1)
            gate = torch.sigmoid(self.gate(gate_in).squeeze(-1))
            delta = torch.tanh(self.delta_head(factor_target).squeeze(-1)) * cap * gate
        return {
            "reason_visual_logits": reason_visual_logits,
            "reason_tfc_delta": delta,
            "reason_logits": reason_visual_logits + delta,
            "reason_tfc_gate": gate,
        }
