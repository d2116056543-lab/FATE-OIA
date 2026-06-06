from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits


def _bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float())


def _asl(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits.float(), targets.float(), gamma_pos=0.0, gamma_neg=4.0, clip=0.05)


def diva_caf_loss(outputs: dict[str, Any], y_action: torch.Tensor, y_reason: torch.Tensor, weights: dict[str, float | str] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    weights = weights or {}
    loss_fn_name = str(weights.get("loss_type", "asl")).lower()
    main_loss_fn = _asl if loss_fn_name == "asl" else _bce
    loss_action_fate = main_loss_fn(outputs["z_fate_action_logits"], y_action)
    loss_action_eva = main_loss_fn(outputs["z_eva_action_logits"], y_action)
    loss_action_actor = main_loss_fn(outputs["z_actor_action_logits"], y_action)
    loss_reason = main_loss_fn(outputs["final_reason_logits"], y_reason)
    loss_base_reason = main_loss_fn(outputs.get("base_reason_logits", outputs["final_reason_logits"]), y_reason)
    loss_gate = _bce(outputs["visual_gate"].clamp(1e-5, 1 - 1e-5).logit(), outputs.get("gate_target", torch.zeros_like(outputs["visual_gate"])))
    caf_stats = outputs.get("selected_vs_random_stats", {}) or {}
    loss_caf = caf_stats.get("loss", outputs["z_actor_action_logits"].new_zeros(()))
    if not torch.is_tensor(loss_caf):
        loss_caf = outputs["z_actor_action_logits"].new_tensor(float(loss_caf))
    main_loss = (
        float(weights.get("fate", 1.00)) * loss_action_fate
        + float(weights.get("eva", 0.50)) * loss_action_eva
        + float(weights.get("actor", 1.00)) * loss_action_actor
        + float(weights.get("reason", 1.00)) * loss_reason
        + float(weights.get("base_reason", 0.25)) * loss_base_reason
    )
    aux_loss = float(weights.get("gate", 0.10)) * loss_gate + float(weights.get("caf", 0.10)) * loss_caf
    total = main_loss + aux_loss
    terms = {
        "loss_action_fate": loss_action_fate,
        "loss_action_eva": loss_action_eva,
        "loss_action_actor": loss_action_actor,
        "loss_reason": loss_reason,
        "loss_base_reason": loss_base_reason,
        "loss_gate": loss_gate,
        "loss_caf_selected_vs_random": loss_caf,
        "main_loss": main_loss,
        "aux_loss": aux_loss,
        "total_loss": total,
        "loss_type": loss_fn_name,
    }
    return total, terms
