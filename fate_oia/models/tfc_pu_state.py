from __future__ import annotations

import torch
from torch import nn


class TFCPUStateBuilder(nn.Module):
    def __init__(self, max_hard_negative_rate: float = 0.20) -> None:
        super().__init__()
        self.max_hard_negative_rate = float(max_hard_negative_rate)

    def forward(
        self,
        reason_targets: torch.Tensor | None,
        credit_reason: torch.Tensor,
        factor_probs: torch.Tensor,
        factor_rho: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor | dict]:
        b, _, reason_dim = credit_reason.shape
        device = credit_reason.device
        dtype = credit_reason.dtype
        if reason_targets is None:
            reason_targets = torch.zeros(b, reason_dim, device=device, dtype=dtype)
        reason_targets = reason_targets.to(device=device, dtype=dtype)
        positive_mask = reason_targets > 0.5
        support_credit = credit_reason.clamp_min(0).sum(dim=1)
        contra_credit = (-credit_reason.clamp_max(0)).sum(dim=1)
        rho_reason = (factor_probs * factor_rho).mean(dim=1, keepdim=True).expand(b, reason_dim)
        unknown_mask = ~positive_mask
        soft_negative_weight = torch.zeros_like(reason_targets)
        hard_negative_mask = torch.zeros_like(positive_mask)
        if epoch >= 3:
            soft_negative_weight = (contra_credit.detach() / (support_credit.detach() + contra_credit.detach() + 1e-6)).clamp(0, 0.4)
            soft_negative_weight = soft_negative_weight * (~positive_mask).float()
        if epoch >= 7:
            candidate = (~positive_mask) & (support_credit.detach() < 0.05) & (contra_credit.detach() > 0.10) & (rho_reason.detach() > 0.20)
            max_count = max(1, int(self.max_hard_negative_rate * reason_dim))
            hard_negative_mask = torch.zeros_like(candidate)
            for i in range(b):
                idx = torch.nonzero(candidate[i], as_tuple=False).flatten()
                if idx.numel() > 0:
                    scores = contra_credit[i, idx]
                    keep = idx[torch.argsort(scores, descending=True)[:max_count]]
                    hard_negative_mask[i, keep] = True
            unknown_mask = unknown_mask & (~hard_negative_mask)
        stats = {
            "positive_count": float(positive_mask.float().sum().detach().cpu()),
            "unknown_count": float(unknown_mask.float().sum().detach().cpu()),
            "soft_negative_weight_sum": float(soft_negative_weight.sum().detach().cpu()),
            "hard_negative_count": float(hard_negative_mask.float().sum().detach().cpu()),
            "hard_negative_by_reason": [float(x) for x in hard_negative_mask.float().sum(0).detach().cpu()],
        }
        return {
            "positive_mask": positive_mask,
            "unknown_mask": unknown_mask,
            "soft_negative_weight": soft_negative_weight,
            "hard_negative_mask": hard_negative_mask,
            "support_credit": support_credit,
            "contra_credit": contra_credit,
            "rho_reason": rho_reason,
            "stats": stats,
        }
