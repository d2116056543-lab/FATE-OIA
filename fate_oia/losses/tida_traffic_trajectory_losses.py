from __future__ import annotations

import torch
from torch.nn import functional as F


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


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
    return _weighted_mean(correction, weight) + 0.5 * _weighted_mean(no_harm, confident + 1e-4)


def trajectory_selected_control_loss(
    base_logits: torch.Tensor,
    selected_delta: torch.Tensor,
    control_delta: torch.Tensor,
    target: torch.Tensor,
    trajectory_support: torch.Tensor,
    *,
    margin_fraction: float = 0.10,
    trajectory_trust: torch.Tensor | None = None,
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
    weight = boundary * (0.10 + 0.90 * trajectory_support.detach().clamp(0.0, 1.0))
    trust = (
        torch.ones_like(trajectory_support)
        if trajectory_trust is None else trajectory_trust.detach().clamp(0.0, 1.0)
    )
    reachable_margin = (
        float(margin_fraction) * float(trajectory_cap)
        * trajectory_support.detach().clamp(0.0, 1.0) * trust
    )
    violation = F.relu(reachable_margin + control_margin - selected_margin)
    return _weighted_mean(violation, weight)
