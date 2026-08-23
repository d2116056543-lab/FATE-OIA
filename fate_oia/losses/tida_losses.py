from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .asymmetric_loss import asymmetric_loss_with_logits
from .tida_flow_credit_losses import (
    conditional_credit_weight,
    conditional_no_harm_weight,
    counterfactual_margin_credit_loss,
    image_fallback_no_harm_loss,
    positive_label_no_harm_loss,
    temporal_utility_calibration_loss,
    transition_alignment_loss,
)
from .tida_traffic_trajectory_losses import (
    trajectory_boundary_correction_loss,
    trajectory_selected_control_loss,
    trajectory_utility_calibration_loss,
)


def terminal_gain_loss(error_history: torch.Tensor, error_no_history: torch.Tensor, margin: float = 0.03) -> torch.Tensor:
    return F.relu(error_history - error_no_history.detach() + float(margin)).mean()


def terminal_order_loss(real_error: torch.Tensor, counterfactual_error: torch.Tensor, margin: float = 0.03) -> torch.Tensor:
    return F.relu(real_error - counterfactual_error + float(margin)).mean()


def action_macro_asl_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raw = asymmetric_loss_with_logits(logits, target.float(), gamma_neg=4, gamma_pos=0, clip=0.05, reduction="none")
    return raw.mean(0).mean()


def action_smooth_ap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    reference_logits: torch.Tensor | None = None,
    reference_target: torch.Tensor | None = None,
    temperature: float = 0.10,
) -> torch.Tensor:
    reference_logits = None if reference_logits is None else reference_logits.detach()
    reference_target = None if reference_target is None else reference_target.detach()
    losses = []
    for label in range(logits.shape[1]):
        positive = logits[target[:, label] > 0.5, label]
        negative = logits[target[:, label] <= 0.5, label]
        reference_positive = logits.new_empty(0)
        if reference_logits is not None and reference_target is not None:
            reference_positive = reference_logits[reference_target[:, label] > 0.5, label]
            reference_negative = reference_logits[reference_target[:, label] <= 0.5, label]
            negative = torch.cat([negative, reference_negative])
        terms = []
        if positive.numel() and negative.numel():
            terms.append(torch.sigmoid((negative[:, None] - positive[None]) / temperature).mean())
        current_negative = logits[target[:, label] <= 0.5, label]
        if reference_positive.numel() and current_negative.numel():
            terms.append(torch.sigmoid((current_negative[:, None] - reference_positive[None]) / temperature).mean())
        if terms:
            losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def action_base_protect_loss(
    image_logits: torch.Tensor,
    video_logits: torch.Tensor,
    target: torch.Tensor,
    reliability: torch.Tensor,
    epsilon: float = 0.005,
) -> torch.Tensor:
    sign = 2.0 * target.float() - 1.0
    image_margin = sign * image_logits
    video_margin = sign * video_logits
    rho = reliability.mean(-1, keepdim=True).detach()
    return ((1.0 - rho) * F.relu(image_margin - video_margin - float(epsilon))).mean(0).mean()


def _target_motion_weight(motion_energy: torch.Tensor, num_targets: int) -> torch.Tensor:
    if motion_energy.ndim == 2:
        motion = motion_energy.detach().mean(1, keepdim=True).expand(-1, num_targets)
        scale = 0.05
    elif motion_energy.ndim == 3 and motion_energy.shape[-1] == num_targets:
        motion = motion_energy.detach().mean(1)
        scale = 0.10
    else:
        raise ValueError("motion_energy must be [B,T] or [B,T,num_targets]")
    return 0.25 + 0.75 * torch.tanh(motion / scale)


def target_conditioned_geometric_correction_loss(
    base_logits: torch.Tensor,
    geometric_delta: torch.Tensor,
    target: torch.Tensor,
    motion_energy: torch.Tensor,
    label_weight: torch.Tensor | None = None,
    *,
    target_margin: float = 0.20,
) -> torch.Tensor:
    """Correct low-margin dynamic examples while preserving confident base decisions."""
    sign = 2.0 * target.float() - 1.0
    base_margin = sign * base_logits.detach()
    final_margin = sign * (base_logits.detach() + geometric_delta)
    need = torch.sigmoid((0.75 - base_margin) / 0.25).detach()
    motion_weight = _target_motion_weight(motion_energy, base_logits.shape[1])
    weight = need * motion_weight
    if label_weight is not None:
        weight = weight * label_weight.detach()
    correction = F.softplus((float(target_margin) - final_margin) / 0.20) * 0.20
    correction = (correction * weight).sum() / weight.sum().clamp_min(1.0)
    confident = torch.sigmoid((base_margin - 0.50) / 0.20).detach()
    protect = (F.relu(base_margin - final_margin - 0.002) * confident).sum() / confident.sum().clamp_min(1.0)
    return correction + 0.5 * protect


def target_conditioned_geometric_ranking_loss(
    base_logits: torch.Tensor,
    geometric_delta: torch.Tensor,
    target: torch.Tensor,
    motion_energy: torch.Tensor,
    label_weight: torch.Tensor | None = None,
    reference_logits: torch.Tensor | None = None,
    reference_target: torch.Tensor | None = None,
    *,
    margin: float = 0.10,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Improve hard positive-negative ordering using only motion-conditioned residuals."""
    final_logits = base_logits.detach() + geometric_delta
    target_motion_weight = _target_motion_weight(motion_energy, base_logits.shape[1])
    losses = []
    for label in range(base_logits.shape[1]):
        sample_motion_weight = target_motion_weight[:, label]
        positive_mask = target[:, label] > 0.5
        negative_mask = ~positive_mask
        positive = final_logits[positive_mask, label]
        negative = final_logits[negative_mask, label]
        base_positive = base_logits.detach()[positive_mask, label]
        base_negative = base_logits.detach()[negative_mask, label]
        terms = []
        if positive.numel() and negative.numel():
            hardness = torch.sigmoid((base_negative[:, None] - base_positive[None] + 0.30) / 0.15)
            pair_motion = 0.5 * (
                sample_motion_weight[negative_mask, None] + sample_motion_weight[None, positive_mask]
            )
            pair_weight = hardness * pair_motion
            if label_weight is not None:
                pair_weight = pair_weight * label_weight.detach()[negative_mask, label, None]
            raw = F.softplus((float(margin) + negative[:, None] - positive[None]) / float(temperature))
            terms.append(float(temperature) * (raw * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8))
        if reference_logits is not None and reference_target is not None:
            reference_positive = reference_logits.detach()[reference_target[:, label] > 0.5, label]
            reference_negative = reference_logits.detach()[reference_target[:, label] <= 0.5, label]
            if positive.numel() and reference_negative.numel():
                raw = F.softplus(
                    (float(margin) + reference_negative[:, None] - positive[None]) / float(temperature)
                )
                terms.append(float(temperature) * (raw * sample_motion_weight[positive_mask][None]).mean())
            if negative.numel() and reference_positive.numel():
                raw = F.softplus(
                    (float(margin) + negative[:, None] - reference_positive[None]) / float(temperature)
                )
                current_weight = sample_motion_weight[negative_mask]
                if label_weight is not None:
                    current_weight = current_weight * label_weight.detach()[negative_mask, label]
                terms.append(float(temperature) * (raw * current_weight[:, None]).mean())
        if terms:
            losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean() if losses else geometric_delta.sum() * 0.0


def action_route_sparse_loss(
    route: torch.Tensor,
    factor_keys: torch.Tensor,
    valid_rho: torch.Tensor,
    eps: float = 1e-8,
    min_diversity_mass: float = 0.01,
    cosine_eps: float = 1e-4,
) -> torch.Tensor:
    entropy = -(route * route.clamp_min(eps).log()).sum(-1) / math.log(route.shape[-1])
    nonnull_mass = 1.0 - route[..., -1]

    # The null key has no semantic direction. Computing cosine diversity from a
    # nearly all-null route makes the centroid norm approach zero and amplifies
    # its gradient by 1 / ||centroid||. Only compare action centroids after they
    # carry enough non-null mass, and use a scale-aware stabilized cosine.
    nonnull_route = route[..., :-1]
    normalized_keys = F.normalize(factor_keys[:, :-1], dim=-1)
    conditional_route = nonnull_route / nonnull_mass.clamp_min(float(min_diversity_mass))[..., None]
    centroids = torch.einsum("baf,bfd->bad", conditional_route, normalized_keys)
    centroid_dot = torch.einsum("bad,bcd->bac", centroids, centroids)
    centroid_sq = centroids.square().sum(-1)
    denominator = (
        (centroid_sq + float(cosine_eps))[:, :, None]
        * (centroid_sq + float(cosine_eps))[:, None, :]
    ).sqrt()
    similarity = centroid_dot / denominator
    actions = route.shape[1]
    off_diagonal = ~torch.eye(actions, dtype=torch.bool, device=route.device)[None]
    centroid_valid = nonnull_mass >= float(min_diversity_mass)
    diversity_mask = off_diagonal & centroid_valid[:, :, None] & centroid_valid[:, None, :]
    diversity_values = F.relu(similarity - 0.90)
    diversity = (diversity_values * diversity_mask.to(diversity_values.dtype)).sum() / diversity_mask.sum().clamp_min(1)
    per_sample = entropy.mean(-1) + F.relu(0.05 - nonnull_mass).mean(-1)
    mask = valid_rho.to(route.dtype)
    route_term = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
    return route_term + diversity


def reason_pu_weight(
    target: torch.Tensor,
    contradiction_scores: torch.Tensor | None = None,
    *,
    negative_floor: float = 0.2,
) -> torch.Tensor:
    contradiction = torch.zeros_like(target) if contradiction_scores is None else contradiction_scores.detach().clamp(0, 1)
    negative = float(negative_floor) + (1.0 - float(negative_floor)) * contradiction
    return torch.where(target > 0.5, torch.ones_like(target), negative).detach()


def reason_partial_asl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    contradiction_scores: torch.Tensor | None = None,
    negative_floor: float = 0.2,
) -> torch.Tensor:
    raw = asymmetric_loss_with_logits(logits, target.float(), gamma_neg=4, gamma_pos=0, clip=0.05, reduction="none")
    weights = reason_pu_weight(target, contradiction_scores, negative_floor=negative_floor)
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def reason_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    negative_weight: torch.Tensor | None = None,
    margin: float = 0.2,
) -> torch.Tensor:
    weights = torch.ones_like(target) if negative_weight is None else negative_weight.detach()
    losses = []
    for label in range(logits.shape[1]):
        positive = logits[target[:, label] > 0.5, label]
        negative_mask = target[:, label] <= 0.5
        negative = logits[negative_mask, label]
        negative_weights = weights[negative_mask, label]
        if positive.numel() and negative.numel():
            raw = F.relu(float(margin) - positive[:, None] + negative[None])
            pair_weight = negative_weights[None].expand_as(raw)
            losses.append((raw * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def reason_soft_f1_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    contradiction_scores: torch.Tensor | None = None,
    negative_floor: float = 0.2,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    weights = reason_pu_weight(target, contradiction_scores, negative_floor=negative_floor)
    true_positive = (probabilities * target * weights).sum(0)
    false_positive = (probabilities * (1.0 - target) * weights).sum(0)
    false_negative = ((1.0 - probabilities) * target * weights).sum(0)
    return 1.0 - ((2.0 * true_positive + 1e-6) / (2.0 * true_positive + false_positive + false_negative + 1e-6)).mean()


def build_tida_loss_registry(
    output: dict[str, torch.Tensor],
    action_target: torch.Tensor,
    reason_target: torch.Tensor,
    *,
    counterfactual_errors: dict[str, torch.Tensor] | None = None,
    counterfactual_outputs: dict[str, dict[str, torch.Tensor]] | None = None,
    rank_reference: dict[str, torch.Tensor] | None = None,
    weights: dict[str, float] | None = None,
    deploy_action_boundary_logits: torch.Tensor | None = None,
):
    from .tida_loss_registry import TIDALossRegistry

    registry = TIDALossRegistry(weights)
    history_error = output["terminal_error_history"]
    no_history_error = output["terminal_error_no_history"]
    registry.add("terminal_hist", history_error.mean())
    registry.add("terminal_no_history", no_history_error.mean())
    registry.add("terminal_gain", terminal_gain_loss(history_error, no_history_error))
    counterfactual_errors = counterfactual_errors or {}
    for name, key in (("temporal_order", "order"), ("repeated_last_contrast", "repeat")):
        if key in counterfactual_errors:
            registry.add(name, terminal_order_loss(history_error, counterfactual_errors[key]))
        else:
            registry.add(
                name,
                history_error.sum() * 0.0,
                available=False,
                unavailable_reason="counterfactual scheduled every four optimizer updates",
            )
    registry.add(
        "flow_transition_align",
        transition_alignment_loss(
            output["transition_tokens"],
            output["predicate_innovation_token"],
            output["transition_reliability"] * output["predicate_innovation_reliability"],
        ),
    )
    registry.add("action_asl", action_macro_asl_loss(output["video_action_logits"], action_target))
    rank_reference = rank_reference or {}
    registry.add(
        "action_smooth_ap",
        action_smooth_ap_loss(
            output["video_action_logits"],
            action_target,
            rank_reference.get("action_logits"),
            rank_reference.get("action_target"),
        ),
    )
    registry.add(
        "action_base_protect",
        action_base_protect_loss(
            output["image_action_logits"], output["video_action_logits"], action_target, output["innovation_reliability"]
        ),
    )
    registry.add("action_delta", output["action_temporal_delta"].square().mean())
    registry.add(
        "action_route_sparse",
        action_route_sparse_loss(
            output["action_route"],
            output["action_factor_keys"],
            valid_rho=output["innovation_reliability"].max(-1).values > 0,
        ),
    )
    counterfactual_outputs = counterfactual_outputs or {}
    action_need = output.get("action_temporal_need", torch.ones_like(action_target))
    action_credit_weight = conditional_credit_weight(action_need)
    action_no_harm_weight = conditional_no_harm_weight(action_need)
    action_credit = [
        counterfactual_margin_credit_loss(
            output["video_action_logits"], value["video_action_logits"], action_target,
            sample_weight=action_credit_weight, margin=0.02
        )
        for value in counterfactual_outputs.values()
    ]
    registry.add(
        "action_flow_credit",
        torch.stack(action_credit).mean() if action_credit else output["video_action_logits"].sum() * 0.0,
        available=bool(action_credit),
        unavailable_reason=None if action_credit else "counterfactual evaluated at optimizer boundary only",
    )
    registry.add(
        "action_flow_no_harm",
        image_fallback_no_harm_loss(
            output["image_action_logits"], output["video_action_logits"], action_target,
            sample_weight=action_no_harm_weight,
        ),
    )
    action_utility = [
        temporal_utility_calibration_loss(
            output["action_temporal_budget"], output["video_action_logits"],
            value["video_action_logits"], action_target,
            max_budget=0.60,
        )
        for value in counterfactual_outputs.values()
    ]
    registry.add(
        "action_utility_calibration",
        torch.stack(action_utility).mean() if action_utility else output["video_action_logits"].sum() * 0.0,
        available=bool(action_utility),
        unavailable_reason=None if action_utility else "counterfactual evaluated at optimizer boundary only",
    )
    registry.add(
        "geometric_action_aux",
        target_conditioned_geometric_correction_loss(
            output["semantic_video_action_logits"], output["geometric_action_delta_raw"],
            action_target, output["geometric_motion_energy"],
        ),
    )
    registry.add(
        "geometric_action_rank",
        target_conditioned_geometric_ranking_loss(
            output["semantic_video_action_logits"], output["geometric_action_delta_raw"],
            action_target, output["geometric_motion_energy"], None,
            rank_reference.get("action_logits"), rank_reference.get("action_target"),
        ),
    )
    prefix_action = output["geometric_prefix_action_logits_raw"].flatten(0, 1)
    prefix_action_target = action_target[:, None].expand(-1, 4, -1).flatten(0, 1)
    registry.add("geometric_action_prefix", action_macro_asl_loss(prefix_action, prefix_action_target))
    registry.add("geometric_action_delta", output["geometric_action_delta_raw"].square().mean())
    registry.add(
        "traffic_action_aux",
        target_conditioned_geometric_correction_loss(
            output["semantic_video_action_logits"], output["traffic_action_delta_raw"],
            action_target, output["traffic_patch_effective_motion"],
        ),
    )
    registry.add(
        "traffic_action_rank",
        target_conditioned_geometric_ranking_loss(
            output["semantic_video_action_logits"], output["traffic_action_delta_raw"],
            action_target, output["traffic_patch_effective_motion"], None,
            rank_reference.get("action_logits"), rank_reference.get("action_target"),
        ),
    )
    registry.add("traffic_action_delta", output["traffic_action_delta_raw"].square().mean())
    trajectory_motion = output["trajectory_speed"].mean(2).permute(0, 2, 1)
    registry.add(
        "trajectory_action_boundary",
        trajectory_boundary_correction_loss(
            output["semantic_video_action_logits"], output["traffic_trajectory_delta_raw"],
            action_target, output["traffic_trajectory_support"],
            deploy_boundary_logits=deploy_action_boundary_logits,
        ),
    )
    registry.add(
        "trajectory_action_rank",
        target_conditioned_geometric_ranking_loss(
            output["semantic_video_action_logits"], output["traffic_trajectory_delta_raw"],
            action_target, trajectory_motion, output["traffic_trajectory_support"],
            rank_reference.get("action_logits"), rank_reference.get("action_target"),
        ),
    )
    trajectory_controls = [
        trajectory_selected_control_loss(
            output["semantic_video_action_logits"], output["traffic_trajectory_delta_raw"],
            output["traffic_trajectory_control_delta_raw"], action_target,
            output["traffic_trajectory_support"],
            trajectory_trust=output["traffic_trajectory_trust"],
            trajectory_order_gate=output["trajectory_order_gate"],
            trajectory_uncertainty_gate=output["trajectory_uncertainty_gate"],
            trajectory_cap=0.08,
            deploy_boundary_logits=deploy_action_boundary_logits,
        )
    ] + [
        trajectory_selected_control_loss(
            output["semantic_video_action_logits"], output["traffic_trajectory_delta_raw"],
            value["traffic_trajectory_delta_raw"], action_target,
            output["traffic_trajectory_support"],
            trajectory_trust=output["traffic_trajectory_trust"],
            trajectory_order_gate=output["trajectory_order_gate"],
            trajectory_uncertainty_gate=output["trajectory_uncertainty_gate"],
            trajectory_cap=0.08,
            deploy_boundary_logits=deploy_action_boundary_logits,
        )
        for value in counterfactual_outputs.values()
    ]
    registry.add(
        "trajectory_selected_control",
        torch.stack(trajectory_controls).mean()
        if trajectory_controls else output["traffic_trajectory_delta_raw"].sum() * 0.0,
        available=bool(trajectory_controls),
        unavailable_reason=None if trajectory_controls else "counterfactual evaluated at optimizer boundary only",
    )
    registry.add(
        "trajectory_utility_calibration",
        trajectory_utility_calibration_loss(
            output["traffic_trajectory_utility_logit"],
            output["traffic_trajectory_candidate_delta"],
            action_target,
        ),
    )
    registry.add("trajectory_delta", output["traffic_trajectory_delta_raw"].square().mean())
    image_branch = output.get("image_branch", {})
    contradiction = image_branch.get("contradiction_score") if isinstance(image_branch, dict) else None
    reason_weights = reason_pu_weight(reason_target, contradiction)
    reason_need = output.get("reason_temporal_need", torch.ones_like(reason_target))
    reason_credit_weight = reason_weights * conditional_credit_weight(reason_need)
    reason_no_harm_weight = reason_weights * conditional_no_harm_weight(reason_need)
    registry.add("reason_partial", reason_partial_asl_loss(output["video_reason_logits"], reason_target, contradiction))
    registry.add(
        "reason_rank",
        reason_rank_loss(output["video_reason_logits"], reason_target, reason_weights),
    )
    registry.add("reason_soft_f1", reason_soft_f1_loss(output["video_reason_logits"], reason_target, contradiction))
    registry.add("reason_delta", output["reason_temporal_delta"].square().mean())
    reason_credit = [
        counterfactual_margin_credit_loss(
            output["video_reason_logits"], value["video_reason_logits"], reason_target,
            sample_weight=reason_credit_weight, margin=0.015,
        )
        for value in counterfactual_outputs.values()
    ]
    registry.add(
        "reason_flow_credit",
        torch.stack(reason_credit).mean() if reason_credit else output["video_reason_logits"].sum() * 0.0,
        available=bool(reason_credit),
        unavailable_reason=None if reason_credit else "counterfactual evaluated at optimizer boundary only",
    )
    registry.add(
        "reason_flow_no_harm",
        image_fallback_no_harm_loss(
            output["image_reason_logits"], output["video_reason_logits"], reason_target,
            sample_weight=reason_no_harm_weight,
        ),
    )
    registry.add(
        "reason_positive_no_harm",
        positive_label_no_harm_loss(
            output["image_reason_logits"], output["video_reason_logits"], reason_target,
        ),
    )
    reason_utility = [
        temporal_utility_calibration_loss(
            output["reason_temporal_budget"], output["video_reason_logits"],
            value["video_reason_logits"], reason_target, max_budget=0.50,
        )
        for value in counterfactual_outputs.values()
    ]
    registry.add(
        "reason_utility_calibration",
        torch.stack(reason_utility).mean() if reason_utility else output["video_reason_logits"].sum() * 0.0,
        available=bool(reason_utility),
        unavailable_reason=None if reason_utility else "counterfactual evaluated at optimizer boundary only",
    )
    registry.add(
        "geometric_reason_aux",
        target_conditioned_geometric_correction_loss(
            output["semantic_video_reason_logits"], output["geometric_reason_delta_raw"],
            reason_target, output["geometric_motion_energy"], reason_weights, target_margin=0.15,
        ),
    )
    registry.add(
        "geometric_reason_rank",
        target_conditioned_geometric_ranking_loss(
            output["semantic_video_reason_logits"], output["geometric_reason_delta_raw"],
            reason_target, output["geometric_motion_energy"], reason_weights,
        ),
    )
    prefix_reason = output["geometric_prefix_reason_logits_raw"].flatten(0, 1)
    prefix_reason_target = reason_target[:, None].expand(-1, 4, -1).flatten(0, 1)
    prefix_contradiction = None if contradiction is None else contradiction[:, None].expand(-1, 4, -1).flatten(0, 1)
    registry.add(
        "geometric_reason_prefix",
        reason_partial_asl_loss(prefix_reason, prefix_reason_target, prefix_contradiction),
    )
    registry.add("geometric_reason_delta", output["geometric_reason_delta_raw"].square().mean())
    return registry
