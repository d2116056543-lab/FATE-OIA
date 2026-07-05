from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.models.tfc_prototype_bank import prototype_consistency_loss


def action_asl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def action_rank_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pos = probs.unsqueeze(2)
    neg = probs.unsqueeze(1)
    mask = (targets.unsqueeze(2) > 0.5) & (targets.unsqueeze(1) < 0.5)
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    return F.relu(0.1 - (pos - neg))[mask].mean()


def action_safe_loss(action_final_logits: torch.Tensor, action_visual_logits: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
    return 0.1 * F.l1_loss(action_final_logits, action_visual_logits.detach())


def reason_pu_asl_loss(reason_logits: torch.Tensor, reason_targets: torch.Tensor, pu_state: dict) -> torch.Tensor:
    y = reason_targets.float()
    bce = F.binary_cross_entropy_with_logits(reason_logits, y, reduction="none")
    pos = pu_state["positive_mask"].float()
    soft_neg = pu_state["soft_negative_weight"].float()
    hard_neg = pu_state["hard_negative_mask"].float()
    weight = pos + soft_neg + hard_neg
    if float(weight.sum().detach().cpu()) <= 0:
        return reason_logits.new_tensor(0.0)
    return (bce * weight).sum() / weight.sum().clamp_min(1.0)


def factor_measurement_loss(factor_probs_action: torch.Tensor, factor_probs_reason: torch.Tensor) -> torch.Tensor:
    qa = factor_probs_action.clamp(1e-4, 1 - 1e-4)
    qr = factor_probs_reason.clamp(1e-4, 1 - 1e-4)
    entropy = -(qa * qa.log() + (1 - qa) * (1 - qa).log()).mean()
    entropy = entropy + (-(qr * qr.log() + (1 - qr) * (1 - qr).log()).mean())
    return 0.01 * entropy


def target_credit_sign_loss(
    credit_action: torch.Tensor,
    credit_reason: torch.Tensor,
    action_targets: torch.Tensor,
    reason_targets: torch.Tensor,
    compatibility: dict[str, torch.Tensor],
    pu_state: dict,
) -> torch.Tensor:
    native_action = (
        compatibility["factor_to_action_support"] - compatibility["factor_to_action_inhibit"]
    ).to(credit_action.device, credit_action.dtype)
    native_reason = (
        compatibility["factor_to_reason_support"] - compatibility["factor_to_reason_inhibit"]
    ).to(credit_reason.device, credit_reason.dtype)
    action_pos = action_targets[:, None, :].float() > 0.5
    reason_pos = pu_state["positive_mask"][:, None, :].float() > 0.5
    hard_reason_neg = pu_state["hard_negative_mask"][:, None, :].float() > 0.5
    action_support = native_action[None, :, :] > 0
    action_inhibit = native_action[None, :, :] < 0
    reason_support = native_reason[None, :, :] > 0
    reason_inhibit = native_reason[None, :, :] < 0
    losses: list[torch.Tensor] = []
    mask = action_pos & action_support
    if bool(mask.any()):
        losses.append(F.softplus(-credit_action[mask]).mean())
    mask = action_pos & action_inhibit
    if bool(mask.any()):
        losses.append(F.softplus(credit_action[mask]).mean())
    mask = reason_pos & reason_support
    if bool(mask.any()):
        losses.append(F.softplus(-credit_reason[mask]).mean())
    mask = reason_pos & reason_inhibit
    if bool(mask.any()):
        losses.append(F.softplus(credit_reason[mask]).mean())
    mask = hard_reason_neg & reason_support
    if bool(mask.any()):
        losses.append(F.softplus(credit_reason[mask]).mean())
    return sum(losses) if losses else credit_action.new_tensor(0.0)


def deletion_contrast_loss(selected_effect: torch.Tensor, random_effect: torch.Tensor) -> torch.Tensor:
    return F.relu(0.02 - (selected_effect - random_effect)).mean()


def calalign_softf1_loss(action_deploy: torch.Tensor, reason_deploy: torch.Tensor, action_targets: torch.Tensor, reason_targets: torch.Tensor, pu_state: dict) -> torch.Tensor:
    return action_asl_loss(action_deploy, action_targets) + reason_pu_asl_loss(reason_deploy, reason_targets, pu_state)


def rate_cardinality_loss(action_logits: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.sigmoid(action_logits).sum(1), action_targets.float().sum(1))


def threshold_smooth_loss(theta_delta_action: torch.Tensor, theta_delta_reason: torch.Tensor) -> torch.Tensor:
    return 0.01 * (theta_delta_action.pow(2).mean() + theta_delta_reason.pow(2).mean())


def compute_tfc_losses(out: dict, action_targets: torch.Tensor, reason_targets: torch.Tensor, weights: dict) -> dict[str, torch.Tensor]:
    la = action_asl_loss(out["action_logits_deploy"], action_targets)
    lar = action_rank_loss(out["action_logits_deploy"], action_targets)
    lsafe = action_safe_loss(out["action_logits_base"], out["action_visual_logits"], action_targets)
    lr = reason_pu_asl_loss(out["reason_logits_deploy"], reason_targets, out["pu_state"])
    lf = factor_measurement_loss(out["factor_probs_action"], out["factor_probs_reason"])
    lp_action, _ = prototype_consistency_loss(
        out["factor_features_action"],
        out["factor_queries"],
        out["factor_prototypes"],
        out["native_similarity"],
        out["factor_conflict"],
    )
    lp_reason, _ = prototype_consistency_loss(
        out["factor_features_reason"],
        out["factor_queries"],
        out["factor_prototypes"],
        out["native_similarity"],
        out["factor_conflict"],
    )
    lproto = 0.5 * (lp_action + lp_reason)
    lc = target_credit_sign_loss(
        out["credit_action"],
        out["credit_reason"],
        action_targets,
        reason_targets,
        out["compatibility"],
        out["pu_state"],
    )
    ld_action = out.get("deletion_stats_action", out["deletion_stats"]).get("deletion_contrast_loss", out["action_logits_deploy"].new_tensor(0.0))
    ld_reason = out.get("deletion_stats_reason", {}).get("deletion_contrast_loss", out["action_logits_deploy"].new_tensor(0.0))
    ld = 0.5 * (ld_action + ld_reason)
    lcal = calalign_softf1_loss(out["action_logits_deploy"], out["reason_logits_deploy"], action_targets, reason_targets, out["pu_state"])
    lsmooth = threshold_smooth_loss(out["theta_delta_action"], out["theta_delta_reason"])
    lcard = rate_cardinality_loss(out["action_logits_deploy"], action_targets)
    total = (
        weights.get("action_asl", 1.0) * la
        + weights.get("action_rank", 0.15) * lar
        + weights.get("reason_pu", 1.0) * lr
        + weights.get("factor_measurement", 0.25) * lf
        + weights.get("prototype_consistency", 0.05) * lproto
        + weights.get("target_credit", 0.10) * lc
        + weights.get("deletion_contrast", 0.05) * ld
        + weights.get("calalign", 0.50) * (lcal + lsmooth)
        + weights.get("rate_cardinality", 0.05) * lcard
        + weights.get("action_safe", 0.50) * lsafe
    )
    return {
        "total": total,
        "action": la + lar,
        "reason": lr,
        "factor": lf + lproto,
        "prototype": lproto,
        "credit": lc,
        "deletion": ld,
        "calalign": lcal + lsmooth,
        "cardinality": lcard,
        "action_safe": lsafe,
        "action_asl": la,
        "action_rank": lar,
    }
