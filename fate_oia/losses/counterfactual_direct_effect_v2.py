from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _valid_positive_mask(cf: dict[str, Any], reason_gt: torch.Tensor) -> torch.Tensor:
    valid = cf.get("cf_real_evidence_mask")
    if not isinstance(valid, torch.Tensor):
        valid = torch.zeros_like(reason_gt, dtype=torch.bool)
    return valid.to(reason_gt.device) & (reason_gt > 0.5)


def direct_effect_loss_v2(
    cf: dict[str, Any],
    reason_gt: torch.Tensor,
    action_gt: torch.Tensor | None = None,
    tail_labels: tuple[int, ...] = (),
    common_margin: float = 0.20,
    tail_margin: float = 0.32,
) -> tuple[torch.Tensor, dict[str, float]]:
    factual = cf["reason_logits_factual"]
    deleted = cf["reason_logits_target_deleted"]
    mask = _valid_positive_mask(cf, reason_gt)
    if not bool(mask.any()):
        z = factual.new_zeros(())
        return z, {
            "cf_valid_count": 0.0,
            "cf_real_evidence_count": 0.0,
            "cf_loss_nonzero_rate": 0.0,
            "direct_effect_mean": 0.0,
            "direct_effect_tail_mean": 0.0,
            "target_deleted_drop_mean": 0.0,
        }
    drop = (torch.sigmoid(factual) - torch.sigmoid(deleted))
    margins = factual.new_full(factual.shape, float(common_margin))
    for r in tail_labels:
        if 0 <= int(r) < margins.shape[1]:
            margins[:, int(r)] = float(tail_margin)
    loss_terms = F.relu(margins - drop)[mask]
    loss = loss_terms.mean()
    tail_mask = torch.zeros_like(mask)
    for r in tail_labels:
        if 0 <= int(r) < mask.shape[1]:
            tail_mask[:, int(r)] = mask[:, int(r)]
    tail_drop = drop[tail_mask].mean() if bool(tail_mask.any()) else factual.new_zeros(())
    return loss, {
        "cf_valid_count": float(mask.sum().detach().cpu()),
        "cf_real_evidence_count": float(mask.sum().detach().cpu()),
        "cf_loss_nonzero_rate": float((loss_terms > 1e-8).float().mean().detach().cpu()),
        "direct_effect_mean": float(drop[mask].mean().detach().cpu()),
        "direct_effect_tail_mean": float(tail_drop.detach().cpu()),
        "target_deleted_drop_mean": float(drop[mask].mean().detach().cpu()),
    }


def evidence_sufficiency_loss_v2(cf: dict[str, Any], reason_gt: torch.Tensor, margin: float = 0.08) -> torch.Tensor:
    mask = _valid_positive_mask(cf, reason_gt)
    if not bool(mask.any()):
        return reason_gt.new_zeros(())
    factual = torch.sigmoid(cf["reason_logits_factual"])
    only = torch.sigmoid(cf["reason_logits_evidence_only"])
    return F.relu(float(margin) - (only - factual.detach() + float(margin)))[mask].mean()


def context_suppression_loss_v2(cf: dict[str, Any], reason_gt: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    mask = _valid_positive_mask(cf, reason_gt)
    if not bool(mask.any()):
        return reason_gt.new_zeros(())
    context = torch.sigmoid(cf["reason_logits_context_only"])
    return F.relu(context - float(margin))[mask].mean()


def non_target_preserve_loss_v2(cf: dict[str, Any], reason_gt: torch.Tensor) -> torch.Tensor:
    valid = cf.get("cf_valid_mask")
    if not isinstance(valid, torch.Tensor):
        return reason_gt.new_zeros(())
    non_target = (~valid.to(reason_gt.device)) & (reason_gt < 0.5)
    if not bool(non_target.any()):
        return reason_gt.new_zeros(())
    factual = torch.sigmoid(cf["reason_logits_factual"]).detach()
    deleted = torch.sigmoid(cf["reason_logits_target_deleted"])
    return F.mse_loss(deleted[non_target], factual[non_target])


def replacement_loss_v2(cf: dict[str, Any], reason_gt: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    mask = _valid_positive_mask(cf, reason_gt)
    if not bool(mask.any()) or "reason_logits_replaced" not in cf:
        z = reason_gt.new_zeros(())
        return z, {"replacement_drop_mean": 0.0, "replacement_valid_count": 0.0}
    factual = torch.sigmoid(cf["reason_logits_factual"])
    replaced = torch.sigmoid(cf["reason_logits_replaced"])
    drop = factual - replaced
    return F.relu(0.05 - drop[mask]).mean(), {
        "replacement_drop_mean": float(drop[mask].mean().detach().cpu()),
        "replacement_valid_count": float(mask.sum().detach().cpu()),
    }


def counterfactual_v2_losses(
    cf: dict[str, Any],
    reason_gt: torch.Tensor,
    action_gt: torch.Tensor | None,
    tail_labels: tuple[int, ...],
    common_margin: float = 0.20,
    tail_margin: float = 0.32,
    context_margin: float = 0.10,
    sufficiency_margin: float = 0.08,
) -> tuple[torch.Tensor, dict[str, float]]:
    direct, stats = direct_effect_loss_v2(cf, reason_gt, action_gt, tail_labels, common_margin, tail_margin)
    context = context_suppression_loss_v2(cf, reason_gt, context_margin)
    preserve = non_target_preserve_loss_v2(cf, reason_gt)
    suff = evidence_sufficiency_loss_v2(cf, reason_gt, sufficiency_margin)
    repl, repl_stats = replacement_loss_v2(cf, reason_gt)
    total = direct + context + preserve + suff + repl
    stats.update(
        {
            "context_suppression_loss": float(context.detach().cpu()),
            "non_target_preserve_loss": float(preserve.detach().cpu()),
            "evidence_sufficiency_loss": float(suff.detach().cpu()),
            "replacement_loss": float(repl.detach().cpu()),
            **repl_stats,
        }
    )
    valid = _valid_positive_mask(cf, reason_gt)
    if bool(valid.any()):
        non_target_drop = torch.sigmoid(cf["reason_logits_factual"]) - torch.sigmoid(cf["reason_logits_target_deleted"])
        not_valid = (~valid) & (reason_gt < 0.5)
        stats["non_target_drop_mean"] = float(non_target_drop[not_valid].mean().detach().cpu()) if bool(not_valid.any()) else 0.0
    else:
        stats["non_target_drop_mean"] = 0.0
    return total, stats
