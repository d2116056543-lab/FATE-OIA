from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def action_delta_cap(epoch: int, max_cap: float = 0.06) -> float:
    if epoch <= 5:
        return 0.0
    if epoch <= 8:
        return min(max_cap, 0.02)
    if epoch <= 10:
        return min(max_cap, 0.04)
    return min(max_cap, 0.06)


class TFCActionHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, max_delta: float = 0.06) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_delta = float(max_delta)
        self.action_queries = nn.Parameter(torch.randn(action_dim, dim) * 0.02)
        self.visual_proj = nn.Linear(dim, dim)
        self.visual_head = nn.Linear(dim, 1)
        self.delta_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.gate = nn.Sequential(nn.Linear(3, 8), nn.GELU(), nn.Linear(8, 1))

    def visual_logits_from_patch(self, patch_action: torch.Tensor) -> torch.Tensor:
        patch = patch_action.mean(1)
        q = F.normalize(self.action_queries, dim=-1)
        k = F.normalize(self.visual_proj(patch), dim=-1)
        attn = torch.softmax(torch.einsum("ad,bnd->ban", q, k), dim=-1)
        target_features = torch.einsum("ban,bnd->bad", attn, patch)
        return self.visual_head(target_features).squeeze(-1)

    def forward(
        self,
        patch_action: torch.Tensor,
        factor_features_action: torch.Tensor,
        credit_action_norm: torch.Tensor,
        credit_confidence_action: torch.Tensor,
        deletion_stats: dict | None,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        action_visual_logits = self.visual_logits_from_patch(patch_action)
        factor_target = torch.einsum("bfa,bfd->bad", credit_action_norm, factor_features_action)
        cap = action_visual_logits.new_tensor(action_delta_cap(epoch, self.max_delta))
        if float(cap.detach().cpu()) <= 0.0:
            delta = torch.zeros_like(action_visual_logits)
            gate = torch.zeros_like(action_visual_logits)
        else:
            deletion_gap = torch.zeros_like(action_visual_logits)
            selected_mask = torch.ones_like(action_visual_logits, dtype=torch.bool)
            if deletion_stats is not None:
                deletion_gap = deletion_stats.get("selected_vs_random_gap", deletion_gap).detach()
                selected_mask = deletion_stats.get("selected_gt_random_mask", selected_mask).detach().bool()
            visual_margin = action_visual_logits.detach().abs()
            gate_in = torch.stack([credit_confidence_action.detach(), deletion_gap, visual_margin], dim=-1)
            gate = torch.sigmoid(self.gate(gate_in).squeeze(-1)) * selected_mask.float()
            delta = torch.tanh(self.delta_head(factor_target).squeeze(-1)) * cap * gate
        return {
            "action_visual_logits": action_visual_logits,
            "action_tfc_delta": delta,
            "action_logits": action_visual_logits + delta,
            "action_tfc_gate": gate,
            "action_target_features": factor_target,
        }
