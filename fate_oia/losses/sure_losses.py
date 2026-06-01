from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel


def make_sure_criterion(loss_name: str = "asl", gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
    if loss_name == "asl":
        return AsymmetricLossMultiLabel(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    return torch.nn.BCEWithLogitsLoss()


def compute_sure_losses(
    outputs: dict[str, Any],
    action_targets: torch.Tensor,
    reason_targets: torch.Tensor,
    criterion,
    relation_teacher_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    action_loss = criterion(outputs["action_final_logits"], action_targets.float())
    reason_loss = criterion(outputs["reason_final_logits"], reason_targets.float())
    base_action_loss = criterion(outputs["action_base_logits"], action_targets.float())
    base_reason_loss = criterion(outputs["reason_base_logits"], reason_targets.float())
    relation_scores = outputs.get("relation_scores")
    if relation_scores is not None:
        target = torch.ones_like(relation_scores)
        relation_teacher = F.binary_cross_entropy_with_logits(relation_scores, target) * float(relation_teacher_weight)
    else:
        relation_teacher = action_loss.new_zeros(())
    return {
        "action": action_loss,
        "reason": reason_loss,
        "base_action": base_action_loss,
        "base_reason": base_reason_loss,
        "relation_teacher": relation_teacher,
    }
