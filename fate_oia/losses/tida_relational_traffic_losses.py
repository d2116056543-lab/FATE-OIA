from __future__ import annotations

import torch
from torch.nn import functional as F


def relational_deletion_contrast_loss(
    full_delta: torch.Tensor,
    selected_deleted_delta: torch.Tensor,
    random_deleted_delta: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    *,
    element_weight: torch.Tensor | None = None,
    margin: float = 0.002,
) -> torch.Tensor:
    """Make target-selected trajectories more necessary than random tracks."""
    if not (
        full_delta.shape == selected_deleted_delta.shape == random_deleted_delta.shape
        == target.shape == support.shape
    ):
        raise ValueError("relational deletion tensors must have identical [B,L] shapes")
    sign = 2.0 * target.float() - 1.0
    selected_damage = sign * (full_delta - selected_deleted_delta)
    random_damage = sign * (full_delta - random_deleted_delta)
    if float(margin) <= 0:
        raise ValueError("margin must be positive")
    # The margin is expressed in logit units (~1e-3). Normalize the hinge so
    # its configured weight remains comparable to classification/ranking
    # losses instead of silently shrinking by three orders of magnitude.
    loss = F.relu(float(margin) + random_damage - selected_damage) / float(margin)
    weight = support.detach().clamp(0.0, 1.0)
    if element_weight is not None:
        if element_weight.shape != target.shape:
            raise ValueError("element_weight must match target")
        weight = weight * element_weight.detach().clamp(0.0, 1.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def relational_proper_no_harm_loss(
    base_logits: torch.Tensor,
    delta: torch.Tensor,
    target: torch.Tensor,
    *,
    element_weight: torch.Tensor | None = None,
    margin: float = 0.002,
) -> torch.Tensor:
    """Penalize target-wise proper-loss regressions against a detached fallback."""
    if not (base_logits.shape == delta.shape == target.shape):
        raise ValueError("relational no-harm tensors must have identical [B,L] shapes")
    base_loss = F.binary_cross_entropy_with_logits(
        base_logits.detach(), target.float(), reduction="none"
    )
    candidate_loss = F.binary_cross_entropy_with_logits(
        base_logits.detach() + delta, target.float(), reduction="none"
    )
    if float(margin) <= 0:
        raise ValueError("margin must be positive")
    regression = F.relu(candidate_loss - base_loss) / float(margin)
    if element_weight is None:
        return regression.mean()
    if element_weight.shape != target.shape:
        raise ValueError("element_weight must match target")
    weight = element_weight.detach().clamp(0.0, 1.0)
    return (regression * weight).sum() / weight.sum().clamp_min(1e-8)
