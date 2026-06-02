from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits


def evidence_bag_loss(outputs: dict[str, Any], reason_targets: torch.Tensor, active_mask: torch.Tensor | None = None, margin: float = 0.05) -> torch.Tensor:
    scores = outputs["evidence_scores"]
    random_scores = outputs["random_evidence_scores"].detach()
    pos = reason_targets.float()
    if active_mask is not None:
        pos = pos * active_mask.float()
    pos_loss = F.softplus(-(scores - random_scores - margin)) * pos
    neg_mask = (1.0 - reason_targets.float()) * (scores.detach() > random_scores.detach()).float()
    neg_loss = F.softplus(scores - random_scores + margin) * neg_mask
    denom = (pos.sum() + neg_mask.sum()).clamp_min(1.0)
    return (pos_loss.sum() + 0.25 * neg_loss.sum()) / denom


def care_moe_training_loss(outputs: dict[str, Any], action: torch.Tensor, reason: torch.Tensor, args: Any) -> tuple[torch.Tensor, dict[str, float]]:
    action_loss = asymmetric_loss_with_logits(outputs["action_logits"], action, gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    reason_loss = asymmetric_loss_with_logits(outputs["reason_logits"], reason, gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    base_action_loss = asymmetric_loss_with_logits(outputs["action_base_logits"], action, gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    base_reason_loss = asymmetric_loss_with_logits(outputs["reason_base_logits"], reason, gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    bag = evidence_bag_loss(outputs, reason, outputs.get("active_reason_mask")) * float(args.loss_evidence_bag)
    reason_delta_reg = outputs["reason_delta"].pow(2).mean() * float(args.loss_reason_delta_reg)
    action_delta_reg = outputs["action_delta"].pow(2).mean() * float(args.loss_action_delta_reg)
    loss = action_loss + reason_loss + 0.2 * base_action_loss + 0.2 * base_reason_loss + bag + reason_delta_reg + action_delta_reg
    parts = {
        "action_loss": float(action_loss.detach().cpu()),
        "reason_loss": float(reason_loss.detach().cpu()),
        "base_action_loss": float(base_action_loss.detach().cpu()),
        "base_reason_loss": float(base_reason_loss.detach().cpu()),
        "evidence_bag_loss": float(bag.detach().cpu()),
        "reason_delta_reg": float(reason_delta_reg.detach().cpu()),
        "action_delta_reg": float(action_delta_reg.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
    }
    return loss, parts
