from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def action_rank_trust_region(final_pos: Tensor, final_neg: Tensor, primary_pos: Tensor, primary_neg: Tensor,
                             m_min: float = 0.10, rho: float = 0.90) -> dict[str, Tensor]:
    primary_margin = (primary_pos - primary_neg).detach()
    final_margin = final_pos - final_neg
    repair_mask = primary_margin <= 0
    preserve_mask = primary_margin > float(m_min)
    zero = final_margin.sum() * 0.0
    repair = F.softplus(-final_margin[repair_mask]).mean() if repair_mask.any() else zero
    preserve = F.relu(float(rho) * primary_margin[preserve_mask] - final_margin[preserve_mask]).mean() if preserve_mask.any() else zero
    return {"repair_loss": repair, "preserve_loss": preserve,
            "primary_correct_pair_count": preserve_mask.sum(),
            "final_preserved_pair_rate": ((final_margin[preserve_mask] > 0).float().mean() if preserve_mask.any() else zero),
            "primary_wrong_pair_repair_rate": ((final_margin[repair_mask] > 0).float().mean() if repair_mask.any() else zero),
            "new_pair_inversion_rate": ((final_margin[preserve_mask] <= 0).float().mean() if preserve_mask.any() else zero)}


def labelwise_reason_rank(final_pos: Tensor, final_neg: Tensor, weight: Tensor | None = None) -> Tensor:
    loss = F.softplus(-(final_pos - final_neg))
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)
