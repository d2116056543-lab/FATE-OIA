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


def evidence_view_consistency_loss(output: dict[str, torch.Tensor], mirrored: dict[str, torch.Tensor], mirror_indices: torch.Tensor) -> torch.Tensor:
    batch = mirrored["explicit_evidence_tokens"].shape[0]
    aligned_coordinates = mirrored["evidence_part_coordinates"][:, mirror_indices].clone()
    aligned_coordinates[..., 0] = 1.0 - aligned_coordinates[..., 0]
    terms = (
        F.smooth_l1_loss(output["explicit_evidence_tokens"][:batch], mirrored["explicit_evidence_tokens"][:, mirror_indices]),
        F.smooth_l1_loss(output["evidence_presence_logits"][:batch], mirrored["evidence_presence_logits"][:, mirror_indices]),
        F.smooth_l1_loss(output["evidence_observability_logits"][:batch], mirrored["evidence_observability_logits"][:, mirror_indices]),
        F.smooth_l1_loss(output["evidence_state_logits"][:batch], mirrored["evidence_state_logits"][:, mirror_indices]),
        F.smooth_l1_loss(output["evidence_masks"][:batch], mirrored["evidence_masks"][:, mirror_indices].flip(-1)),
        F.smooth_l1_loss(output["evidence_part_coordinates"][:batch], aligned_coordinates),
    )
    return sum(terms) / len(terms)


def refinement_loss(direct_logits: torch.Tensor, refined_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    direct = F.binary_cross_entropy_with_logits(direct_logits.detach(), targets.float(), reduction="none")
    refined = F.binary_cross_entropy_with_logits(refined_logits, targets.float(), reduction="none")
    hard = (direct >= direct.median()).float()
    easy = 1.0 - hard
    improve = (refined - direct).relu() * hard
    regress = (refined - direct + 0.02).relu() * easy
    return (improve + regress).mean()


def semantic_reliability_weight(
    reason_direct_logits: torch.Tensor,
    reason_semantic_logits: torch.Tensor,
    reason_attention: torch.Tensor,
    evidence_reliability: torch.Tensor,
    evidence_view_consistency: torch.Tensor,
    floor: float = 0.25,
) -> torch.Tensor:
    branch = 1.0 - (torch.sigmoid(reason_direct_logits) - torch.sigmoid(reason_semantic_logits)).abs()
    evidence = torch.einsum("bre,be->br", reason_attention, evidence_reliability).clamp(0.0, 1.0)
    view = torch.einsum("bre,e->br", reason_attention, evidence_view_consistency).clamp(0.0, 1.0)
    return (floor + (1.0 - floor) * (view * branch * evidence).clamp_min(0.0).pow(1.0 / 3.0)).detach().clamp(floor, 1.0)


def evidence_loss(evidence: dict[str, torch.Tensor], targets: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    anchor = evidence["presence_logits"]
    zero = anchor.sum() * 0.0
    if targets is None:
        return {"loss_evidence": zero, "loss_evidence_presence": zero, "loss_evidence_state": zero, "loss_evidence_geometry": zero, "loss_evidence_prototype": zero, "loss_evidence_view": zero, "loss_evidence_latent_diversity": zero, "curve_distance_valid_count": torch.tensor(0, device=anchor.device), "valid_count": torch.tensor(0, device=anchor.device)}
    presence_target = targets.get("presence")
    valid = targets.get("presence_valid")
    if presence_target is None or valid is None or valid.sum() == 0:
        return {"loss_evidence": zero, "loss_evidence_presence": zero, "loss_evidence_state": zero, "loss_evidence_geometry": zero, "loss_evidence_prototype": zero, "loss_evidence_view": zero, "loss_evidence_latent_diversity": zero, "curve_distance_valid_count": torch.tensor(0, device=anchor.device), "valid_count": torch.tensor(0, device=anchor.device)}
    presence = F.binary_cross_entropy_with_logits(anchor, presence_target.float(), reduction="none")
    observability = F.binary_cross_entropy_with_logits(evidence["observability_logits"], targets["observability"].float(), reduction="none")
    presence_value = ((presence + 0.3 * observability) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    state_valid = targets.get("state_valid", torch.zeros_like(valid)).float()
    state_target = targets.get("state", evidence["state_logits"].detach() * 0.0)
    state_raw = F.binary_cross_entropy_with_logits(evidence["state_logits"], state_target.float(), reduction="none").mean(-1)
    state_value = (state_raw * state_valid).sum() / state_valid.sum().clamp_min(1.0)
    part_valid = targets.get("part_valid", torch.zeros_like(valid)).float()
    part_target = targets.get("part_coordinates", evidence["part_coordinates"].detach())
    part_mask = evidence["part_valid"].to(anchor).view(1, *evidence["part_valid"].shape)
    coordinate_raw = F.smooth_l1_loss(evidence["part_coordinates"], part_target.float(), reduction="none").mean(-1)
    coordinate_value = (coordinate_raw * part_mask * part_valid.unsqueeze(-1)).sum() / (part_mask * part_valid.unsqueeze(-1)).sum().clamp_min(1.0)
    scale_target = targets.get("part_scales", evidence["part_scales"].detach()).float()
    scale_raw = F.smooth_l1_loss(evidence["part_scales"], scale_target, reduction="none").mean(-1)
    scale_value = (scale_raw * part_mask * part_valid.unsqueeze(-1)).sum() / (part_mask * part_valid.unsqueeze(-1)).sum().clamp_min(1.0)
    height, width = evidence["soft_masks"].shape[-2:]
    yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, height, device=anchor.device, dtype=anchor.dtype), torch.linspace(0.0, 1.0, width, device=anchor.device, dtype=anchor.dtype), indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).view(1, 1, 1, height, width, 2)
    target_distance = ((grid - part_target.float().unsqueeze(-2).unsqueeze(-2)) / 0.07).square().sum(-1)
    generated_masks = (torch.exp(-0.5 * target_distance) * part_mask.unsqueeze(-1).unsqueeze(-1)).amax(2)
    target_masks = targets.get("soft_masks", generated_masks).to(anchor).float()
    predicted_masks = evidence["soft_masks"].clamp(1e-5, 1.0 - 1e-5)
    geometry_type = evidence["geometry_type"].view(1, -1)
    point = (geometry_type == 0).float() * part_valid
    region = (geometry_type == 1).float() * part_valid
    curve = (geometry_type == 2).float() * part_valid
    focal = F.binary_cross_entropy(predicted_masks, target_masks, reduction="none") * (predicted_masks - target_masks).abs().pow(2)
    focal = (focal.mean((-1, -2)) * point).sum() / point.sum().clamp_min(1.0)
    intersection = (predicted_masks * target_masks).sum((-1, -2))
    dice = 1.0 - (2.0 * intersection + 1e-5) / (predicted_masks.sum((-1, -2)) + target_masks.sum((-1, -2)) + 1e-5)
    region_dice = (dice * region).sum() / region.sum().clamp_min(1.0)
    pred_dx = predicted_masks[..., 1:] - predicted_masks[..., :-1]
    target_dx = target_masks[..., 1:] - target_masks[..., :-1]
    pred_dy = predicted_masks[..., 1:, :] - predicted_masks[..., :-1, :]
    target_dy = target_masks[..., 1:, :] - target_masks[..., :-1, :]
    cldice = (pred_dx - target_dx).abs().mean((-1, -2)) + (pred_dy - target_dy).abs().mean((-1, -2))
    cldice = (cldice * curve).sum() / curve.sum().clamp_min(1.0)
    curve_indices = torch.where(evidence["geometry_type"] == 2)[0]
    curve_distance = zero
    if curve_indices.numel() > 0:
        predicted_curve = evidence["part_coordinates"][:, curve_indices]
        target_curve = part_target[:, curve_indices].float()
        distances = torch.cdist(predicted_curve.flatten(0, 1), target_curve.flatten(0, 1))
        symmetric = distances.min(-1).values.mean(-1) + distances.min(-2).values.mean(-1)
        curve_valid = part_valid[:, curve_indices].flatten()
        curve_distance = (symmetric * curve_valid).sum() / curve_valid.sum().clamp_min(1.0)
    geometry_value = coordinate_value + scale_value + focal + region_dice + cldice + curve_distance
    margin = evidence["prototype_margin"]
    prototype_raw = presence_target.float() * F.softplus(-margin) + (1.0 - presence_target.float()) * F.softplus(margin)
    prototype_value = (prototype_raw * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    latent = F.normalize(evidence["latent_tokens"], dim=-1)
    latent_similarity = torch.einsum("bld,bmd->blm", latent, latent)
    identity = torch.eye(latent.shape[1], device=latent.device, dtype=torch.bool).unsqueeze(0)
    latent_diversity = latent_similarity.masked_select((~identity).expand_as(latent_similarity)).square().mean()
    view_value = evidence.get("view_consistency_loss", zero)
    value = presence_value + 0.3 * state_value + 0.2 * geometry_value + 0.1 * prototype_value + 0.05 * view_value + 0.02 * latent_diversity
    return {"loss_evidence": value, "loss_evidence_presence": presence_value, "loss_evidence_state": state_value, "loss_evidence_geometry": geometry_value, "loss_evidence_prototype": prototype_value, "loss_evidence_view": view_value, "loss_evidence_latent_diversity": latent_diversity, "curve_distance_valid_count": curve.sum(), "valid_count": valid.sum()}


def total_precise_losses(output: dict[str, torch.Tensor], action_targets: torch.Tensor, reason_targets: torch.Tensor, evidence_targets: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    action_final = asymmetric_multilabel_loss(output["action_logits_final_raw"], action_targets)
    action_direct = asymmetric_multilabel_loss(output["action_logits_direct"], action_targets)
    reason_weight = semantic_reliability_weight(
        output["reason_logits_direct"],
        output["reason_logits_semantic"],
        output["reason_evidence_attention"],
        output["evidence_reliability"],
        output["evidence_view_consistency"],
        float(output.get("semantic_weight_floor", 0.25)),
    )
    reason_semantic = asymmetric_multilabel_loss(output["reason_logits_semantic"], reason_targets, reason_weight)
    reason_direct = asymmetric_multilabel_loss(output["reason_logits_direct"], reason_targets)
    reason_observed = asymmetric_multilabel_loss(output["reason_logits_observed"], reason_targets)
    evidence = evidence_loss({"presence_logits": output["evidence_presence_logits"], "observability_logits": output["evidence_observability_logits"], "state_logits": output["evidence_state_logits"], "part_coordinates": output["evidence_part_coordinates"], "part_scales": output["evidence_part_scales"], "soft_masks": output["evidence_masks"], "prototype_margin": output["evidence_prototype_margin"], "latent_tokens": output["latent_evidence_tokens"], "part_valid": output["evidence_part_valid"], "geometry_type": output["evidence_geometry_type"]}, evidence_targets)
    total = action_final + 0.5 * action_direct + reason_semantic + 0.5 * reason_direct + reason_observed + 0.15 * evidence["loss_evidence"]
    return {"loss_total": total, "loss_action_final": action_final, "loss_action_direct": action_direct, "loss_reason_semantic": reason_semantic, "loss_reason_direct": reason_direct, "loss_reason_observed": reason_observed, **evidence}
