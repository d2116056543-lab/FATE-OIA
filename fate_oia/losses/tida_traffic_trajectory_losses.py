from __future__ import annotations

import torch
from torch.nn import functional as F


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def _class_balanced_action_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Average positive/negative evidence equally within each action.

    Traffic corrections are sparse and boundary-focused. A global weighted mean
    otherwise lets the much larger easy class determine the delta orientation.
    Missing classes in a mini-batch are ignored rather than assigned zero loss.
    """
    target = target.float()
    class_means = []
    class_valid = []
    for class_mask in (target, 1.0 - target):
        class_weight = weight * class_mask
        denominator = class_weight.sum(dim=0)
        class_means.append((value * class_weight).sum(dim=0) / denominator.clamp_min(1e-8))
        class_valid.append(denominator > 0)
    means = torch.stack(class_means, dim=0)
    valid = torch.stack(class_valid, dim=0)
    action_mean = (means * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1)
    action_valid = valid.any(dim=0)
    return (action_mean * action_valid).sum() / action_valid.sum().clamp_min(1)


def trajectory_boundary_correction_loss(
    base_logits: torch.Tensor,
    trajectory_delta: torch.Tensor,
    target: torch.Tensor,
    trajectory_support: torch.Tensor,
    *,
    target_margin: float = 0.20,
    deploy_boundary_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use traffic trajectories where the frozen image decision is uncertain."""
    if not (base_logits.shape == trajectory_delta.shape == target.shape == trajectory_support.shape):
        raise ValueError("trajectory boundary tensors must have identical [B,A] shapes")
    sign = 2.0 * target.float() - 1.0
    boundary_logits = (
        torch.zeros(base_logits.shape[-1], device=base_logits.device, dtype=base_logits.dtype)
        if deploy_boundary_logits is None else deploy_boundary_logits.to(base_logits)
    )
    if boundary_logits.shape != (base_logits.shape[-1],):
        raise ValueError("deploy_boundary_logits must be [A]")
    deploy_base = base_logits.detach() - boundary_logits
    base_margin = sign * deploy_base
    final_margin = sign * (deploy_base + trajectory_delta)
    boundary = torch.sigmoid((0.80 - base_margin.abs()) / 0.20).detach()
    support = trajectory_support.detach().clamp(0.0, 1.0)
    weight = boundary * (0.10 + 0.90 * support)
    correction = 0.20 * F.softplus((float(target_margin) - final_margin) / 0.20)
    confident = torch.sigmoid((base_margin - 0.75) / 0.15).detach()
    no_harm = F.relu(base_margin - final_margin - 0.002)
    return _class_balanced_action_mean(correction, weight, target) + 0.5 * _class_balanced_action_mean(
        no_harm, confident + 1e-4, target
    )


def trajectory_selected_control_loss(
    base_logits: torch.Tensor,
    selected_delta: torch.Tensor,
    control_delta: torch.Tensor,
    target: torch.Tensor,
    trajectory_support: torch.Tensor,
    *,
    margin_fraction: float = 0.25,
    trajectory_trust: torch.Tensor | None = None,
    trajectory_order_gate: torch.Tensor | None = None,
    trajectory_uncertainty_gate: torch.Tensor | None = None,
    trajectory_cap: float = 0.08,
    deploy_boundary_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Require ordered selected trajectories to beat same-clip temporal controls."""
    if not (
        base_logits.shape == selected_delta.shape == control_delta.shape
        == target.shape == trajectory_support.shape
    ):
        raise ValueError("selected/control tensors must have identical [B,A] shapes")
    sign = 2.0 * target.float() - 1.0
    boundary_logits = (
        torch.zeros(base_logits.shape[-1], device=base_logits.device, dtype=base_logits.dtype)
        if deploy_boundary_logits is None else deploy_boundary_logits.to(base_logits)
    )
    if boundary_logits.shape != (base_logits.shape[-1],):
        raise ValueError("deploy_boundary_logits must be [A]")
    deploy_base = base_logits.detach() - boundary_logits
    selected_margin = sign * (deploy_base + selected_delta)
    control_margin = sign * (deploy_base + control_delta.detach())
    boundary = torch.sigmoid((0.80 - deploy_base.abs()) / 0.20)
    support = trajectory_support.detach().clamp(0.0, 1.0)
    support_gate = support / (support + 0.05)
    weight = boundary * (0.10 + 0.90 * support_gate)
    trust = (
        torch.ones_like(trajectory_support)
        if trajectory_trust is None else trajectory_trust.detach().clamp(0.0, 1.0)
    )
    order_gate = (
        torch.ones_like(trajectory_support)
        if trajectory_order_gate is None else trajectory_order_gate.detach().clamp(0.0, 1.0)
    )
    uncertainty_gate = (
        torch.ones_like(trajectory_support)
        if trajectory_uncertainty_gate is None
        else trajectory_uncertainty_gate.detach().clamp(0.0, 1.0)
    )
    reachable_margin = (
        float(margin_fraction) * float(trajectory_cap)
        * support_gate * trust * order_gate * uncertainty_gate
    )
    violation = F.relu(reachable_margin + control_margin - selected_margin)
    return _class_balanced_action_mean(violation, weight, target)


def trajectory_utility_calibration_loss(
    utility_logits: torch.Tensor,
    candidate_delta: torch.Tensor,
    target: torch.Tensor,
    *,
    state_utility_logits: torch.Tensor | None = None,
    state_candidate_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Teach an inference-only gate to retain helpful trajectory candidates.

    The correctness target is detached and label-derived only inside this loss;
    labels never enter the model forward or the deployed utility gate.
    """
    if not (utility_logits.shape == candidate_delta.shape == target.shape):
        raise ValueError("trajectory utility tensors must have identical [B,A] shapes")
    sign = 2.0 * target.float() - 1.0
    helpful = (sign * candidate_delta.detach() > 0).to(utility_logits.dtype)
    confidence = (candidate_delta.detach().abs() / 0.02).clamp(0.0, 1.0)
    value = F.binary_cross_entropy_with_logits(utility_logits, helpful, reduction="none")
    order_loss = _class_balanced_action_mean(value, 0.25 + 0.75 * confidence, helpful)
    if state_utility_logits is None and state_candidate_delta is None:
        return order_loss
    if state_utility_logits is None or state_candidate_delta is None:
        raise ValueError("state utility logits and candidate delta must be provided together")
    if not (state_utility_logits.shape == state_candidate_delta.shape == target.shape):
        raise ValueError("state utility tensors must have identical [B,A] shapes")
    state_helpful = (sign * state_candidate_delta.detach() > 0).to(state_utility_logits.dtype)
    state_confidence = (state_candidate_delta.detach().abs() / 0.005).clamp(0.0, 1.0)
    state_value = F.binary_cross_entropy_with_logits(
        state_utility_logits, state_helpful, reduction="none"
    )
    state_loss = _class_balanced_action_mean(
        state_value, 0.25 + 0.75 * state_confidence, state_helpful
    )
    return 0.5 * (order_loss + state_loss)
