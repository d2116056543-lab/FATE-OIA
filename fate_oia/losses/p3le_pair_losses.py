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


def p3le_pair_loss(outputs: dict[str, torch.Tensor], action: torch.Tensor, reason: torch.Tensor, args, return_tensors: bool = False):
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

    epoch = int(float(outputs.get("epoch_tensor", final_action.new_tensor(0.0)).detach().cpu()))
    pair_targets = build_pair_seed_targets(action, reason, action.shape[1], reason.shape[1])
    q_pair = q.detach().unsqueeze(1)
    pair_targets = pair_targets * q_pair
    pair_weights = 0.25 + 0.75 * q_pair
    tail_indices = getattr(args, "tail_indices", [])
    if tail_indices:
        tail_weight = torch.ones_like(pair_weights)
        valid_tail = [idx for idx in tail_indices if idx < tail_weight.shape[-1]]
        if valid_tail:
            tail_weight[:, :, valid_tail] = 0.5
        pair_weights = pair_weights * tail_weight
    pair_bce = F.binary_cross_entropy_with_logits(outputs["pair_tensor"], pair_targets, reduction="none")
    evidence_active = outputs["evidence_lambda_active"].detach()
    pair_loss = (pair_bce * pair_weights).mean() * (0.5 + 0.5 * evidence_active)
    pair_consistency = (
        F.mse_loss(outputs["pair_action_support"].sigmoid(), action.float())
        + F.mse_loss(outputs["pair_reason_support"].sigmoid(), reason.float())
    )
    pair_stage_active = 1.0 if epoch >= 5 else 0.0
    action_set_loss = F.binary_cross_entropy_with_logits(outputs["action_set_logits"], action.float())
    evidence_loss = outputs["evidence_loss"]
    q_entropy = -(q.clamp(1e-6, 1 - 1e-6) * q.clamp(1e-6, 1 - 1e-6).log()).mean()
    gate_entropy = outputs.get("gate_entropy", q.new_zeros(()))
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
        + pair_stage_active * float(args.loss_pair_seed) * pair_loss
        + pair_stage_active * float(args.loss_pair_consistency) * pair_consistency
        + pair_stage_active * float(args.loss_evidence_bag) * evidence_loss
        + float(args.loss_q_entropy) * q_entropy
        + float(args.loss_pareto) * (pareto_action + pareto_reason)
        - float(getattr(args, "loss_gate_entropy", 0.001)) * gate_entropy
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
        "pair_stage_active": float(pair_stage_active),
        "pair_consistency_loss": float(pair_consistency.detach().cpu()),
        "action_set_loss": float(action_set_loss.detach().cpu()),
        "evidence_bag_loss": float(evidence_loss.detach().cpu()),
        "q_entropy": float(q_entropy.detach().cpu()),
        "pareto_action_loss": float(pareto_action.detach().cpu()),
        "pareto_reason_loss": float(pareto_reason.detach().cpu()),
        "router_scale": float(outputs["router_scale"].detach().cpu()),
        "action_router_scale": float(outputs.get("action_router_scale", outputs["router_scale"]).detach().cpu()),
        "reason_router_scale": float(outputs.get("reason_router_scale", outputs["router_scale"]).detach().cpu()),
        "action_gate_mean": float(outputs["action_router_gate"].detach().mean().cpu()),
        "reason_gate_mean": float(outputs["reason_router_gate"].detach().mean().cpu()),
        "action_gate_entropy": float(outputs.get("action_gate_entropy", gate_entropy).detach().cpu()),
        "reason_gate_entropy": float(outputs.get("reason_gate_entropy", gate_entropy).detach().cpu()),
        "gate_entropy": float(gate_entropy.detach().cpu()),
        "q_mean": float(q.detach().mean().cpu()),
        "q_min": float(q.detach().min().cpu()),
        "q_max": float(q.detach().max().cpu()),
        "pair_tensor_mean": float(outputs["pair_tensor"].detach().mean().cpu()),
        "pair_tensor_std": float(outputs["pair_tensor"].detach().std(unbiased=False).cpu()),
        "evidence_selected_mean": float(outputs["evidence_selected_mean"].detach().cpu()),
        "evidence_random_mean": float(outputs["evidence_random_mean"].detach().cpu()),
        "evidence_lambda_active": float(outputs["evidence_lambda_active"].detach().cpu()),
        "bdd100k_prior_positive_rate": float(outputs["bdd100k_prior_positive_rate"].detach().cpu()),
        "pair_sparse_action_topk": float(outputs.get("action_sparse_topk", final_action.new_zeros(())).detach().cpu()),
        "pair_sparse_reason_topk": float(outputs.get("reason_sparse_topk", final_action.new_zeros(())).detach().cpu()),
        "pair_sparse_action_weight_mean": float(outputs.get("action_sparse_weight_mean", final_action.new_zeros(())).detach().cpu()),
        "pair_sparse_reason_weight_mean": float(outputs.get("reason_sparse_weight_mean", final_action.new_zeros(())).detach().cpu()),
        "action_prototype_usage_max": float(outputs.get("action_prototype_usage", final_action.new_zeros(1)).detach().max().cpu()),
        "action_prototype_usage_entropy": float((-(outputs.get("action_prototype_usage", final_action.new_ones(1) / 1.0).detach().clamp(1e-6, 1.0) * outputs.get("action_prototype_usage", final_action.new_ones(1) / 1.0).detach().clamp(1e-6, 1.0).log()).sum(dim=-1).mean()).cpu()) if "action_prototype_usage" in outputs else 0.0,
    }
    if return_tensors:
        tensor_parts = {
            "action_task_loss": action_loss + float(args.loss_a_action) * a_action_loss + float(args.loss_action_set) * action_set_loss,
            "reason_task_loss": reason_loss + float(args.loss_r_reason) * r_reason_loss,
            "pair_task_loss": pair_stage_active * (float(args.loss_pair_seed) * pair_loss + float(args.loss_pair_consistency) * pair_consistency),
        }
        return total, parts, tensor_parts
    return total, parts
