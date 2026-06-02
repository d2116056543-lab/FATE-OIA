from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.losses.care_moe_losses import evidence_bag_loss


def _asl(logits: torch.Tensor, target: torch.Tensor, args: Any) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, target, gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)


def care_act_training_loss(outputs: dict[str, Any], action: torch.Tensor, reason: torch.Tensor, args: Any) -> tuple[torch.Tensor, dict[str, float]]:
    action_loss = _asl(outputs["action_final_candidate_logits"], action, args)
    reason_loss = _asl(outputs["reason_logits"], reason, args)
    base_action_loss = _asl(outputs["action_base_logits"], action, args) * float(args.loss_base_action)
    base_reason_loss = _asl(outputs["reason_base_logits"], reason, args) * float(args.loss_base_reason)
    action_visual_loss = _asl(outputs.get("action_visual_logits", outputs["action_base_logits"]), action, args) * float(args.loss_action_visual)
    r2a_loss = _asl(outputs.get("reason_to_action_logits", outputs["action_base_logits"]), action, args) * float(args.loss_r2a_gt)
    action_evidence_loss = _asl(outputs["action_evidence_logits"], action, args) * float(args.loss_action_evidence)
    action_set_loss = _asl(outputs["action_set_logits"], action, args) * float(args.loss_action_set)
    visual_prob = torch.sigmoid(outputs.get("action_visual_logits", outputs["action_base_logits"]))
    reason_prob = torch.sigmoid(outputs.get("reason_to_action_logits", outputs["action_base_logits"]))
    agree = F.mse_loss(visual_prob, reason_prob) * float(args.loss_action_agree)
    bag = evidence_bag_loss(outputs, reason, outputs.get("active_reason_mask")) * float(args.loss_evidence_bag)
    reason_delta_reg = outputs["reason_delta"].pow(2).mean() * float(args.loss_reason_delta_reg)
    action_delta_reg = outputs["action_total_delta"].pow(2).mean() * float(args.loss_action_delta_reg)
    loss = action_loss + reason_loss + base_action_loss + base_reason_loss + action_visual_loss + r2a_loss + agree + action_evidence_loss + action_set_loss + bag + reason_delta_reg + action_delta_reg
    parts = {
        "action_loss": float(action_loss.detach().cpu()),
        "reason_loss": float(reason_loss.detach().cpu()),
        "base_action_loss": float(base_action_loss.detach().cpu()),
        "base_reason_loss": float(base_reason_loss.detach().cpu()),
        "action_visual_loss": float(action_visual_loss.detach().cpu()),
        "reason_to_action_gt_loss": float(r2a_loss.detach().cpu()),
        "action_agreement_loss": float(agree.detach().cpu()),
        "action_evidence_loss": float(action_evidence_loss.detach().cpu()),
        "action_set_loss": float(action_set_loss.detach().cpu()),
        "evidence_bag_loss": float(bag.detach().cpu()),
        "reason_delta_reg": float(reason_delta_reg.detach().cpu()),
        "action_delta_reg": float(action_delta_reg.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
    }
    return loss, parts
