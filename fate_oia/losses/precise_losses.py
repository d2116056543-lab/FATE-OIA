from __future__ import annotations

import torch
from torch.nn import functional as F


def asymmetric_multilabel_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None, gamma_negative: float = 2.0, gamma_positive: float = 0.0) -> torch.Tensor:
    targets = targets.float()
    probs = torch.sigmoid(logits)
    positive = -targets * (1.0 - probs).pow(gamma_positive) * F.logsigmoid(logits)
    negative = -(1.0 - targets) * probs.pow(gamma_negative) * F.logsigmoid(-logits)
    value = positive + negative
    if weight is not None:
        value = value * weight.detach()
    return value.mean()


def two_way_consistency_loss(logits: torch.Tensor, mirrored_logits: torch.Tensor, mirror_indices: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(logits, mirrored_logits[:, mirror_indices])


def refinement_loss(direct_logits: torch.Tensor, refined_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    direct = F.binary_cross_entropy_with_logits(direct_logits.detach(), targets.float(), reduction="none")
    refined = F.binary_cross_entropy_with_logits(refined_logits, targets.float(), reduction="none")
    hard = (direct >= direct.median()).float()
    easy = 1.0 - hard
    improve = (refined - direct).relu() * hard
    regress = (refined - direct + 0.02).relu() * easy
    return (improve + regress).mean()


def semantic_reliability_weight(reason_direct_logits: torch.Tensor, reason_semantic_logits: torch.Tensor, reason_attention: torch.Tensor, evidence_reliability: torch.Tensor) -> torch.Tensor:
    branch = 1.0 - (torch.sigmoid(reason_direct_logits) - torch.sigmoid(reason_semantic_logits)).abs()
    evidence = torch.einsum("bre,be->br", reason_attention, evidence_reliability).clamp(0.0, 1.0)
    view = torch.ones_like(evidence)
    return (0.25 + 0.75 * (view * branch * evidence).clamp_min(1e-8).pow(1.0 / 3.0)).detach().clamp(0.25, 1.0)


def evidence_loss(evidence: dict[str, torch.Tensor], targets: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    anchor = evidence["presence_logits"]
    zero = anchor.sum() * 0.0
    if targets is None:
        return {"loss_evidence": zero, "valid_count": torch.tensor(0, device=anchor.device)}
    presence_target = targets.get("presence")
    valid = targets.get("presence_valid")
    if presence_target is None or valid is None or valid.sum() == 0:
        return {"loss_evidence": zero, "valid_count": torch.tensor(0, device=anchor.device)}
    presence = F.binary_cross_entropy_with_logits(anchor, presence_target.float(), reduction="none")
    observability = F.binary_cross_entropy_with_logits(evidence["observability_logits"], targets["observability"].float(), reduction="none")
    value = ((presence + 0.3 * observability) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    return {"loss_evidence": value, "valid_count": valid.sum()}


def total_precise_losses(output: dict[str, torch.Tensor], action_targets: torch.Tensor, reason_targets: torch.Tensor, evidence_targets: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    action_final = asymmetric_multilabel_loss(output["action_logits_final_raw"], action_targets)
    action_direct = asymmetric_multilabel_loss(output["action_logits_direct"], action_targets)
    reason_weight = semantic_reliability_weight(output["reason_logits_direct"], output["reason_logits_semantic"], output["reason_evidence_attention"], output["evidence_reliability"])
    reason_semantic = asymmetric_multilabel_loss(output["reason_logits_semantic"], reason_targets, reason_weight)
    reason_direct = asymmetric_multilabel_loss(output["reason_logits_direct"], reason_targets)
    reason_observed = asymmetric_multilabel_loss(output["reason_logits_observed"], reason_targets)
    evidence = evidence_loss({"presence_logits": output["evidence_presence_logits"], "observability_logits": output["evidence_observability_logits"]}, evidence_targets)
    total = action_final + 0.5 * action_direct + reason_semantic + 0.5 * reason_direct + reason_observed + 0.15 * evidence["loss_evidence"]
    return {"loss_total": total, "loss_action_final": action_final, "loss_action_direct": action_direct, "loss_reason_semantic": reason_semantic, "loss_reason_direct": reason_direct, "loss_reason_observed": reason_observed, **evidence}
