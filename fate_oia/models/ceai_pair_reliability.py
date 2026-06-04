from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.ceai_pair_sparse_attention import default_reason_to_group


class PairReliabilityHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, reason_group_count: int = 6, reason_to_group: list[int] | None = None) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.reason_group_count = reason_group_count
        self.reason_to_group = reason_to_group or default_reason_to_group(reason_dim)
        self.support = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.reliability = nn.Sequential(nn.LayerNorm(dim * 3 + 3), nn.Linear(dim * 3 + 3, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, action_tokens: torch.Tensor, reason_tokens: torch.Tensor, pair_group_context: torch.Tensor, base_action_logits: torch.Tensor | None = None, base_reason_logits: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict[str, float]]:
        b, a, d = action_tokens.shape
        reason_contexts = []
        for r, g in enumerate(self.reason_to_group[: self.reason_dim]):
            reason_contexts.append(pair_group_context[:, :, g, :])
        ctx = torch.stack(reason_contexts, dim=2)
        act = action_tokens.unsqueeze(2).expand(-1, -1, self.reason_dim, -1)
        rea = reason_tokens.unsqueeze(1).expand(-1, self.action_dim, -1, -1)
        support_in = torch.cat([act, rea, ctx], dim=-1)
        pair_support = self.support(support_in).squeeze(-1)
        if base_action_logits is None:
            action_unc = torch.zeros(b, self.action_dim, device=action_tokens.device, dtype=action_tokens.dtype)
        else:
            pa = torch.sigmoid(base_action_logits)
            action_unc = 1.0 - (pa - 0.5).abs() * 2.0
        if base_reason_logits is None:
            reason_unc = torch.zeros(b, self.reason_dim, device=action_tokens.device, dtype=action_tokens.dtype)
        else:
            pr = torch.sigmoid(base_reason_logits)
            reason_unc = 1.0 - (pr - 0.5).abs() * 2.0
        aux = torch.stack([
            pair_support,
            action_unc.unsqueeze(2).expand_as(pair_support),
            reason_unc.unsqueeze(1).expand_as(pair_support),
        ], dim=-1)
        q_in = torch.cat([support_in, aux], dim=-1)
        q_ar = torch.sigmoid(self.reliability(q_in).squeeze(-1))
        q_r = q_ar.max(dim=1).values.clamp(0.05, 0.95)
        stats = {
            "q_ar_mean": float(q_ar.detach().mean().cpu()),
            "q_ar_std": float(q_ar.detach().std(unbiased=False).cpu()),
            "q_r_mean": float(q_r.detach().mean().cpu()),
            "pair_support_mean": float(pair_support.detach().mean().cpu()),
            "pair_support_std": float(pair_support.detach().std(unbiased=False).cpu()),
            "pair_reliability_active_rate": float((q_ar.detach() > 0.5).float().mean().cpu()),
        }
        return {"pair_support": pair_support, "pair_reliability": q_ar, "reason_reliability": q_r, "stats": stats}


def build_pair_seed_targets(action_labels: torch.Tensor, reason_labels: torch.Tensor, q_ar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = action_labels.unsqueeze(2) * reason_labels.unsqueeze(1)
    weight = target * q_ar.detach().clamp(0.0, 1.0)
    return target, weight
