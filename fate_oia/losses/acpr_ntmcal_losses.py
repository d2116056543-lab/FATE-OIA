from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.models.acpr_ntmcal_text_atoms import native_text_structure_loss
from fate_oia.losses.acpr_threshold_losses import soft_f1_loss, predicted_positive_rate_loss, action_cardinality_loss


def action_asl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def ntmcal_reason_pu_loss(logits: torch.Tensor, reason_targets: torch.Tensor, pu_state: dict, epoch: int) -> torch.Tensor:
    pos = pu_state["positive_mask"].float()
    # PU reliability is produced by the predicate measurement branch. Detach it here so
    # the reason loss cannot reduce its own penalty by collapsing predicate rho to zero.
    soft_neg = pu_state["soft_negative_weight"].float().detach()
    hard_neg = pu_state["hard_negative_mask"].float().detach()
    weight = pos + soft_neg + hard_neg
    target = pos
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight.sum() <= 0:
        return logits.sum() * 0.0
    return (bce * weight).sum() / weight.sum().clamp_min(1.0)


def native_predicate_measurement_loss(q_pred: torch.Tensor, rho_pred: torch.Tensor, observations: dict, epoch: int) -> torch.Tensor:
    mask = observations["obs_mask"].to(q_pred.device).float()
    value = observations["obs_value"].to(q_pred.device).float()
    soft_neg = observations["obs_soft_negative"].to(q_pred.device).float()
    q = q_pred.float().clamp(1e-5, 1 - 1e-5)
    pos_loss = -(value * q.log() + (1.0 - value) * (1.0 - q).clamp_min(1e-5).log()) * mask
    neg_w = soft_neg if epoch >= 3 else torch.zeros_like(soft_neg)
    neg_loss = -torch.log((1 - q).clamp_min(1e-5)) * neg_w
    rho = rho_pred.float().clamp(1e-5, 1 - 1e-5)
    pos_observed = (mask > 0).float()
    soft_observed = ((soft_neg > 0).float() * (1.0 - pos_observed)).clamp(0, 1)
    unknown = (1.0 - (pos_observed + soft_observed).clamp(0, 1)).clamp(0, 1)
    # Reliability should not be a free escape variable for PU loss, but it also should not
    # saturate to all-one. Keep observed predicates more reliable than unknown predicates
    # while preserving label-wise discrimination for hard-negative mining.
    rho_target = 0.50 * unknown + 0.70 * pos_observed + 0.62 * soft_observed
    rho_weight = 0.25 * unknown + 1.00 * pos_observed + 0.75 * soft_observed
    rho_calibration = ((rho - rho_target).pow(2) * rho_weight).sum() / rho_weight.sum().clamp_min(1.0)
    denom = (mask + neg_w).sum().clamp_min(1.0)
    return (pos_loss + neg_loss).sum() / denom + 0.05 * rho_calibration


def ntmcal_calibration_loss(out: dict, action_targets: torch.Tensor, reason_targets: torch.Tensor) -> torch.Tensor:
    deploy = torch.cat([out["action_logits_deploy"], out["reason_logits_deploy"]], dim=-1)
    targets = torch.cat([action_targets.float(), reason_targets.float()], dim=-1)
    target_rate = targets.mean(0).detach()
    theta = torch.cat([out["theta_action"].mean(0), out["theta_reason"].mean(0)], dim=0)
    teacher = torch.cat([out["theta_action_teacher"], out["theta_reason_teacher"]], dim=0).to(theta.device, theta.dtype)
    return (
        0.35 * F.binary_cross_entropy_with_logits(out["action_logits_deploy"], action_targets.float())
        + 0.35 * ntmcal_reason_pu_loss(out["reason_logits_deploy"], reason_targets, out["pu_state"], 99)
        + 0.15 * soft_f1_loss(out["action_logits_deploy"], action_targets)
        + 0.15 * soft_f1_loss(out["reason_logits_deploy"], reason_targets)
        + 0.10 * predicted_positive_rate_loss(deploy, target_rate)
        + 0.10 * action_cardinality_loss(out["action_logits_deploy"], action_targets)
        + 0.20 * F.smooth_l1_loss(theta, teacher)
    )


def action_predicate_margin_loss(out: dict, action_targets: torch.Tensor, epoch: int) -> torch.Tensor:
    if epoch < 7:
        return out["action_logits_deploy"].sum() * 0.0
    return F.relu(0.1 - out["action_logits_ntmcal"] * (action_targets.float() * 2 - 1)).mean() * out["predicate_q"].mean().detach()


def predicate_attention_sparsity_loss(attn: torch.Tensor) -> torch.Tensor:
    entropy = -(attn.clamp_min(1e-8).log() * attn).sum(-1)
    return entropy.mean()


def schedule_weights(epoch: int) -> dict[str, float]:
    if epoch <= 2:
        return {"action": 1.0, "reason": 1.0, "pred": 0.35, "text": 0.10, "cal": 0.15, "res": 0.0, "act_pred": 0.0, "pair": 0.0, "sparse": 0.01}
    if epoch <= 6:
        return {"action": 1.0, "reason": 1.0, "pred": 0.30, "text": 0.10, "cal": 0.45, "res": 0.05, "act_pred": 0.0, "pair": 0.0, "sparse": 0.01}
    if epoch <= 12:
        return {"action": 1.0, "reason": 1.0, "pred": 0.20, "text": 0.05, "cal": 0.55, "res": 0.05, "act_pred": 0.02, "pair": 0.03, "sparse": 0.005}
    return {"action": 1.0, "reason": 1.0, "pred": 0.10, "text": 0.02, "cal": 0.60, "res": 0.02, "act_pred": 0.02, "pair": 0.01, "sparse": 0.0}


def acpr_ntmcal_loss_bundle(out: dict, action_targets: torch.Tensor, reason_targets: torch.Tensor, epoch: int, cfg: dict) -> tuple[torch.Tensor, dict]:
    w = schedule_weights(epoch)
    l_action = action_asl_loss(out["action_logits_deploy"], action_targets)
    l_reason = ntmcal_reason_pu_loss(out["reason_logits_deploy"], reason_targets, out["pu_state"], epoch)
    l_pred = native_predicate_measurement_loss(out["predicate_q"], out["predicate_rho"], out["native_text_observations"], epoch)
    l_text = native_text_structure_loss(out["_atom_encoder"], out["_predicate_specs"])["native_text_structure_loss"] if "_atom_encoder" in out else out["action_logits_deploy"].sum() * 0.0
    l_cal = ntmcal_calibration_loss(out, action_targets, reason_targets)
    l_act_pred = action_predicate_margin_loss(out, action_targets, epoch)
    l_sparse = predicate_attention_sparsity_loss(out["predicate_topk_attention"])
    main = l_action + l_reason
    l_pair, pair_stats = out["_pair_memory"].loss(out["reason_logits_deploy"], reason_targets, out["pu_state"], epoch, main_loss=main) if "_pair_memory" in out else (main * 0.0, {})
    total = (
        w["action"] * l_action
        + w["reason"] * l_reason
        + w["pred"] * l_pred
        + w["text"] * l_text
        + w["cal"] * l_cal
        + w["act_pred"] * l_act_pred
        + w["pair"] * l_pair
        + w["sparse"] * l_sparse
    )
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_action": float(l_action.detach().cpu()),
        "loss_reason_pu": float(l_reason.detach().cpu()),
        "loss_predicate_measurement": float(l_pred.detach().cpu()),
        "loss_native_text_structure": float(l_text.detach().cpu()),
        "loss_ntmcal_calibration": float(l_cal.detach().cpu()),
        "loss_action_predicate": float(l_act_pred.detach().cpu()),
        "loss_pair": float(l_pair.detach().cpu()),
        "loss_attention_sparsity": float(l_sparse.detach().cpu()),
        **{f"weight_{k}": v for k, v in w.items()},
        **pair_stats,
    }
    return total, stats

