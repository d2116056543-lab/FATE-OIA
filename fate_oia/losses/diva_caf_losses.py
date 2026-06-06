from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float())


def diva_caf_loss(outputs: dict[str, Any], y_action: torch.Tensor, y_reason: torch.Tensor, weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    loss_action_fate = _bce(outputs["z_fate_action_logits"], y_action)
    loss_action_eva = _bce(outputs["z_eva_action_logits"], y_action)
    loss_action_actor = _bce(outputs["z_actor_action_logits"], y_action)
    loss_reason = _bce(outputs["final_reason_logits"], y_reason)
    loss_gate = _bce(outputs["visual_gate"].clamp(1e-5, 1 - 1e-5).logit(), outputs.get("gate_target", torch.zeros_like(outputs["visual_gate"])))
    caf_stats = outputs.get("selected_vs_random_stats", {}) or {}
    loss_caf = caf_stats.get("loss", outputs["z_actor_action_logits"].new_zeros(()))
    if not torch.is_tensor(loss_caf):
        loss_caf = outputs["z_actor_action_logits"].new_tensor(float(loss_caf))
    main_loss = (
        float(weights.get("fate", 0.25)) * loss_action_fate
        + float(weights.get("eva", 0.50)) * loss_action_eva
        + float(weights.get("actor", 1.00)) * loss_action_actor
        + float(weights.get("reason", 1.00)) * loss_reason
    )
    aux_loss = float(weights.get("gate", 0.10)) * loss_gate + float(weights.get("caf", 0.10)) * loss_caf
    total = main_loss + aux_loss
    terms = {
        "loss_action_fate": loss_action_fate,
        "loss_action_eva": loss_action_eva,
        "loss_action_actor": loss_action_actor,
        "loss_reason": loss_reason,
        "loss_gate": loss_gate,
        "loss_caf_selected_vs_random": loss_caf,
        "main_loss": main_loss,
        "aux_loss": aux_loss,
        "total_loss": total,
    }
    return total, terms
