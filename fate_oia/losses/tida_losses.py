from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .asymmetric_loss import asymmetric_loss_with_logits


def terminal_gain_loss(error_history: torch.Tensor, error_no_history: torch.Tensor, margin: float = 0.03) -> torch.Tensor:
    return F.relu(error_history - error_no_history.detach() + float(margin)).mean()


def terminal_order_loss(real_error: torch.Tensor, counterfactual_error: torch.Tensor, margin: float = 0.03) -> torch.Tensor:
    return F.relu(real_error - counterfactual_error + float(margin)).mean()


def action_macro_asl_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raw = asymmetric_loss_with_logits(logits, target.float(), gamma_neg=4, gamma_pos=0, clip=0.05, reduction="none")
    return raw.mean(0).mean()


def action_smooth_ap_loss(logits: torch.Tensor, target: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    losses = []
    for label in range(logits.shape[1]):
        positive = logits[target[:, label] > 0.5, label]
        negative = logits[target[:, label] <= 0.5, label]
        if positive.numel() and negative.numel():
            losses.append(torch.sigmoid((negative[:, None] - positive[None]) / temperature).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def action_base_protect_loss(
    image_logits: torch.Tensor,
    video_logits: torch.Tensor,
    target: torch.Tensor,
    reliability: torch.Tensor,
    epsilon: float = 0.0,
) -> torch.Tensor:
    sign = 2.0 * target.float() - 1.0
    image_margin = sign * image_logits
    video_margin = sign * video_logits
    rho = reliability.mean(-1, keepdim=True).detach()
    return ((1.0 - rho) * F.relu(image_margin - video_margin - float(epsilon))).mean(0).mean()


def action_route_sparse_loss(route: torch.Tensor, factor_keys: torch.Tensor, valid_rho: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    entropy = -(route * route.clamp_min(eps).log()).sum(-1) / math.log(route.shape[-1])
    nonnull_mass = 1.0 - route[..., -1]
    normalized_keys = F.normalize(factor_keys, dim=-1)
    centroids = torch.einsum("baf,bfd->bad", route, normalized_keys)
    centroid_norm = F.normalize(centroids, dim=-1)
    similarity = torch.einsum("bad,bcd->bac", centroid_norm, centroid_norm)
    actions = route.shape[1]
    off_diagonal = ~torch.eye(actions, dtype=torch.bool, device=route.device)[None]
    diversity = F.relu(similarity - 0.90).masked_select(off_diagonal.expand_as(similarity)).mean()
    per_sample = entropy.mean(-1) + F.relu(0.05 - nonnull_mass).mean(-1)
    mask = valid_rho.to(route.dtype)
    route_term = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
    return route_term + diversity


def reason_partial_asl_loss(logits: torch.Tensor, target: torch.Tensor, negative_weight: float = 0.2) -> torch.Tensor:
    raw = asymmetric_loss_with_logits(logits, target.float(), gamma_neg=4, gamma_pos=0, clip=0.05, reduction="none")
    weights = torch.where(target > 0.5, torch.ones_like(target), torch.full_like(target, float(negative_weight)))
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def reason_rank_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    losses = []
    for label in range(logits.shape[1]):
        positive = logits[target[:, label] > 0.5, label]
        negative = logits[target[:, label] <= 0.5, label]
        if positive.numel() and negative.numel():
            losses.append(F.relu(float(margin) - positive[:, None] + negative[None]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def reason_soft_f1_loss(logits: torch.Tensor, target: torch.Tensor, negative_weight: float = 0.2) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    weights = torch.where(target > 0.5, torch.ones_like(target), torch.full_like(target, float(negative_weight)))
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
    weights: dict[str, float] | None = None,
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
    registry.add("action_asl", action_macro_asl_loss(output["video_action_logits"], action_target))
    registry.add("action_smooth_ap", action_smooth_ap_loss(output["video_action_logits"], action_target))
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
    registry.add("reason_partial", reason_partial_asl_loss(output["video_reason_logits"], reason_target))
    registry.add("reason_rank", reason_rank_loss(output["video_reason_logits"], reason_target))
    registry.add("reason_soft_f1", reason_soft_f1_loss(output["video_reason_logits"], reason_target))
    registry.add("reason_delta", output["reason_temporal_delta"].square().mean())
    return registry
