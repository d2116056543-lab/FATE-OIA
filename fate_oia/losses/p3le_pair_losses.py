from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.models.p3le_pair_head import build_pair_seed_targets


def weighted_asl(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, args) -> torch.Tensor:
    loss = asymmetric_loss_with_logits(
        logits,
        targets,
        gamma_pos=float(args.asl_gamma_pos),
        gamma_neg=float(args.asl_gamma_neg),
        clip=float(args.asl_clip),
        reduction="none",
    )
    return (loss * weights.float()).mean()


def p3le_pair_loss(outputs: dict[str, torch.Tensor], action: torch.Tensor, reason: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, float]]:
    final_action = outputs["final_action_logits"]
    final_reason = outputs["final_reason_logits"]
    a_action = outputs["action_specialist_logits"]
    r_reason = outputs["reason_specialist_logits"]
    a_reason = outputs["a_reason_aux_logits"]
    r_action = outputs["r_action_aux_logits"]
    q = outputs["reason_reliability"]

    action_loss = asymmetric_loss_with_logits(final_action, action, args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)
    reason_loss = weighted_asl(final_reason, reason, q.detach(), args)
    a_action_loss = asymmetric_loss_with_logits(a_action, action, args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)
    r_reason_loss = weighted_asl(r_reason, reason, q.detach(), args)
    a_reason_loss = asymmetric_loss_with_logits(a_reason, reason, args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)
    r_action_loss = asymmetric_loss_with_logits(r_action, action, args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)

    pair_targets = build_pair_seed_targets(action, reason, action.shape[1], reason.shape[1])
    pair_loss = F.binary_cross_entropy_with_logits(outputs["pair_tensor"], pair_targets)
    pair_consistency = (
        F.mse_loss(outputs["pair_action_support"].sigmoid(), action.float())
        + F.mse_loss(outputs["pair_reason_support"].sigmoid(), reason.float())
    )
    action_set_loss = F.binary_cross_entropy_with_logits(outputs["action_set_logits"], action.float())
    evidence_loss = outputs["evidence_loss"]
    q_entropy = -(q.clamp(1e-6, 1 - 1e-6) * q.clamp(1e-6, 1 - 1e-6).log()).mean()
    pareto_action = torch.relu(
        asymmetric_loss_with_logits(final_action, action, args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip)
        - a_action_loss.detach()
        + float(args.pareto_margin_action)
    )
    pareto_reason = torch.relu(
        weighted_asl(final_reason, reason, q.detach(), args)
        - r_reason_loss.detach()
        + float(args.pareto_margin_reason)
    )
    total = (
        float(args.loss_action_gt) * action_loss
        + float(args.loss_reason_gt) * reason_loss
        + float(args.loss_a_action) * a_action_loss
        + float(args.loss_r_reason) * r_reason_loss
        + float(args.loss_a_reason) * a_reason_loss
        + float(args.loss_r_action) * r_action_loss
        + float(args.loss_action_set) * action_set_loss
        + float(args.loss_pair_seed) * pair_loss
        + float(args.loss_pair_consistency) * pair_consistency
        + float(args.loss_evidence_bag) * evidence_loss
        + float(args.loss_q_entropy) * q_entropy
        + float(args.loss_pareto) * (pareto_action + pareto_reason)
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "reason_loss": float(reason_loss.detach().cpu()),
        "a_action_loss": float(a_action_loss.detach().cpu()),
        "r_reason_loss": float(r_reason_loss.detach().cpu()),
        "a_reason_loss": float(a_reason_loss.detach().cpu()),
        "r_action_loss": float(r_action_loss.detach().cpu()),
        "pair_seed_loss": float(pair_loss.detach().cpu()),
        "pair_consistency_loss": float(pair_consistency.detach().cpu()),
        "action_set_loss": float(action_set_loss.detach().cpu()),
        "evidence_bag_loss": float(evidence_loss.detach().cpu()),
        "q_entropy": float(q_entropy.detach().cpu()),
        "pareto_action_loss": float(pareto_action.detach().cpu()),
        "pareto_reason_loss": float(pareto_reason.detach().cpu()),
        "router_scale": float(outputs["router_scale"].detach().cpu()),
        "action_gate_mean": float(outputs["action_router_gate"].detach().mean().cpu()),
        "reason_gate_mean": float(outputs["reason_router_gate"].detach().mean().cpu()),
        "q_mean": float(q.detach().mean().cpu()),
        "q_min": float(q.detach().min().cpu()),
        "q_max": float(q.detach().max().cpu()),
        "pair_tensor_mean": float(outputs["pair_tensor"].detach().mean().cpu()),
        "pair_tensor_std": float(outputs["pair_tensor"].detach().std(unbiased=False).cpu()),
        "evidence_selected_mean": float(outputs["evidence_selected_mean"].detach().cpu()),
        "evidence_random_mean": float(outputs["evidence_random_mean"].detach().cpu()),
        "evidence_lambda_active": float(outputs["evidence_lambda_active"].detach().cpu()),
        "bdd100k_prior_positive_rate": float(outputs["bdd100k_prior_positive_rate"].detach().cpu()),
    }
    return total, parts
