from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits


def psr_train_loss(outputs: dict[str, torch.Tensor], action: torch.Tensor, reason: torch.Tensor, args: Any) -> tuple[torch.Tensor, dict[str, float]]:
    gamma_pos = float(getattr(args, "asl_gamma_pos", 0.0))
    gamma_neg = float(getattr(args, "asl_gamma_neg", 4.0))
    clip = float(getattr(args, "asl_clip", 0.05))
    final_action_loss = asymmetric_loss_with_logits(outputs["final_action_logits"], action, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    final_reason_loss = asymmetric_loss_with_logits(outputs["final_reason_logits"], reason, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    a_action_loss = asymmetric_loss_with_logits(outputs["a_action_logits"], action, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    e_reason_loss = asymmetric_loss_with_logits(outputs["e_reason_logits"], reason, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    a_reason_loss = asymmetric_loss_with_logits(outputs["a_reason_logits"], reason, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    e_action_loss = asymmetric_loss_with_logits(outputs["e_action_logits"], action, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    c_reason_loss = asymmetric_loss_with_logits(outputs["c_reason_logits"], reason, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    margin_action = float(getattr(args, "pareto_margin_action", 0.005))
    margin_reason = float(getattr(args, "pareto_margin_reason", 0.005))
    pareto_action = F.relu(final_action_loss - a_action_loss.detach() + margin_action)
    pareto_reason = F.relu(final_reason_loss - e_reason_loss.detach() + margin_reason)
    gate_budget = outputs["reason_router_gate"].mean() * 0.0 + outputs["action_router_gate"].mean() * 0.0
    conflict_budget = outputs.get("shared_conflict_proxy", final_action_loss.new_zeros(()))
    loss = (
        float(getattr(args, "loss_final_action", 1.0)) * final_action_loss
        + float(getattr(args, "loss_final_reason", 1.0)) * final_reason_loss
        + float(getattr(args, "loss_a_action", 0.4)) * a_action_loss
        + float(getattr(args, "loss_e_reason", 0.4)) * e_reason_loss
        + float(getattr(args, "loss_a_reason", 0.05)) * a_reason_loss
        + float(getattr(args, "loss_e_action", 0.01)) * e_action_loss
        + float(getattr(args, "loss_calibration_reason", 0.05)) * c_reason_loss
        + float(getattr(args, "loss_pareto", 0.2)) * (pareto_action + pareto_reason)
        + float(getattr(args, "loss_gradient_budget", 0.001)) * conflict_budget
        + gate_budget
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "final_action_loss": float(final_action_loss.detach().cpu()),
        "final_reason_loss": float(final_reason_loss.detach().cpu()),
        "a_action_loss": float(a_action_loss.detach().cpu()),
        "e_reason_loss": float(e_reason_loss.detach().cpu()),
        "a_reason_loss": float(a_reason_loss.detach().cpu()),
        "e_action_loss": float(e_action_loss.detach().cpu()),
        "c_reason_loss": float(c_reason_loss.detach().cpu()),
        "pareto_action_loss": float(pareto_action.detach().cpu()),
        "pareto_reason_loss": float(pareto_reason.detach().cpu()),
        "gradient_budget_loss": float(conflict_budget.detach().cpu()),
        "router_scale": float(outputs["router_scale"].detach().cpu()),
        "action_gate_mean": float(outputs["action_router_gate"].detach().mean().cpu()),
        "reason_gate_mean": float(outputs["reason_router_gate"].detach().mean().cpu()),
    }
    return loss, parts
