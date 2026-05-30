from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _pos_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum().clamp_min(1.0)
    return (x * mask.float()).sum() / denom


def counterfactual_direct_effect_loss(cf: dict[str, torch.Tensor], reason_gt: torch.Tensor, action_gt: torch.Tensor | None = None, tail_labels: tuple[int, ...] = (), cfg: Any | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    factual = cf["reason_logits_factual"]
    deleted = cf["reason_logits_target_deleted"]
    target = reason_gt.float()
    effect = factual - deleted
    valid = target > 0.5
    loss = F.softplus(0.10 - effect)
    out = _pos_mean(loss, valid)
    stats = {
        "direct_effect_loss": float(out.detach().cpu()),
        "direct_effect_mean": float(_pos_mean(effect.detach(), valid).cpu()) if bool(valid.any()) else 0.0,
        "cf_valid_count": int(valid.sum().item()),
    }
    if tail_labels:
        idx = torch.tensor([i for i in tail_labels if 0 <= i < reason_gt.shape[1]], device=reason_gt.device)
        if idx.numel():
            tail_mask = torch.zeros_like(valid)
            tail_mask[:, idx] = valid[:, idx]
            stats["direct_effect_tail_mean"] = float(_pos_mean(effect.detach(), tail_mask).cpu()) if bool(tail_mask.any()) else 0.0
    return out, stats


def counterfactual_replacement_loss(cf: dict[str, torch.Tensor], reason_gt: torch.Tensor, memory_bank: Any | None = None, cfg: Any | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    replaced = cf.get("reason_logits_replaced")
    if replaced is None:
        return reason_gt.new_zeros(()), {"replacement_false_activation_rate": 0.0}
    neg = reason_gt.float() < 0.5
    false_prob = torch.sigmoid(replaced)
    loss = _pos_mean(F.relu(false_prob - 0.20), neg)
    return loss, {"replacement_false_activation_rate": float(((false_prob > 0.5) & neg).float().mean().detach().cpu())}


def non_target_preservation_loss(factual_logits: torch.Tensor, deleted_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    non_target = (target_mask.float() < 0.5).float()
    return ((factual_logits - deleted_logits).abs() * non_target).sum() / non_target.sum().clamp_min(1.0)


def context_false_positive_rate(cf: dict[str, torch.Tensor], reason_gt: torch.Tensor) -> float:
    ctx = cf.get("reason_logits_context_only")
    if ctx is None:
        return 0.0
    neg = reason_gt.float() < 0.5
    return float(((torch.sigmoid(ctx) > 0.5) & neg).float().mean().detach().cpu())

