from __future__ import annotations

import torch
from torch import Tensor

from .meter_action_losses import asymmetric_multilabel_elements


def dense_factor_intervention_loss(
    action_logits_final: Tensor,
    factor_contributions: Tensor,
    action_target: Tensor,
    *,
    margin: float = 0.02,
) -> dict[str, Tensor]:
    """Dense analytic necessity and target-specificity over every batch."""
    sign = action_target * 2.0 - 1.0
    signed = sign.unsqueeze(-1) * factor_contributions
    correct_index = signed.argmax(-1)
    wrong_index = signed.argmin(-1)
    correct = factor_contributions.gather(
        -1, correct_index.unsqueeze(-1)
    ).squeeze(-1)
    wrong = factor_contributions.gather(-1, wrong_index.unsqueeze(-1)).squeeze(-1)
    base = asymmetric_multilabel_elements(action_logits_final, action_target)
    deleted = asymmetric_multilabel_elements(
        action_logits_final - correct, action_target
    )
    necessity = torch.relu(margin + base - deleted).mean()
    specificity = torch.relu(margin + wrong.abs() - correct.abs()).mean()
    coverage = (factor_contributions.abs() > 1e-6).any(0)
    return {
        "necessity": necessity,
        "specificity": specificity,
        "correct_effect": correct.detach(),
        "wrong_effect": wrong.detach(),
        "action_coverage": coverage.any(-1).sum(),
        "factor_coverage": coverage.any(0).sum(),
        "total": necessity + specificity,
    }


def identity_corruption_loss(
    clean_contributions: Tensor,
    corrupt_contributions: Tensor,
    target: Tensor,
    *,
    margin: float = 0.02,
) -> Tensor:
    if clean_contributions.shape != corrupt_contributions.shape:
        raise ValueError("clean and corrupt identity tensors must have the same shape")
    if clean_contributions.shape == target.shape:
        # Final-route intervention: compare the actual bounded action delta,
        # rather than a proxy sum of raw factor contributions.
        sign = target * 2.0 - 1.0
        clean_score = sign * clean_contributions
        corrupt_score = sign * corrupt_contributions
    elif clean_contributions.shape[:2] == target.shape:
        sign = target.unsqueeze(-1) * 2.0 - 1.0
        # Legacy factor-space path retained for diagnostics that inspect the
        # exact factor decomposition directly.
        clean_score = (sign * clean_contributions).sum(-1)
        corrupt_score = (sign * corrupt_contributions).sum(-1)
    else:
        raise ValueError("identity tensors must be [B,A] or [B,A,F]")
    return torch.relu(margin + corrupt_score - clean_score).mean()


def near_boundary_delta_ranking_loss(
    action_logits_visual: Tensor,
    action_evidence_delta: Tensor,
    target: Tensor,
    *,
    margin: float = 0.02,
    radius: float = 0.75,
) -> Tensor:
    """Reward a target-aligned transport delta most near the decision boundary."""
    sign = target * 2.0 - 1.0
    visual_margin = sign * action_logits_visual.detach()
    boundary_weight = torch.exp(-visual_margin.abs() / float(radius))
    aligned_delta = sign * action_evidence_delta
    penalty = torch.relu(float(margin) - aligned_delta)
    return (boundary_weight * penalty).sum() / boundary_weight.sum().clamp_min(1e-8)


def reason_identity_corruption_loss(
    clean_logits: Tensor,
    corrupt_logits: Tensor,
    target: Tensor,
    *,
    margin: float = 0.02,
) -> Tensor:
    clean = asymmetric_multilabel_elements(clean_logits, target).mean(-1)
    corrupt = asymmetric_multilabel_elements(corrupt_logits, target).mean(-1)
    return torch.relu(margin + clean - corrupt).mean()


def meter_counterfactual_loss(*args: Tensor, **kwargs: Tensor) -> dict[str, Tensor]:
    # Legacy API retained only for old unit-test import compatibility.
    selected_effect, control_effect, wrong_target_effect, support, counter = args[:5]
    selected_control = torch.relu(0.02 - selected_effect + control_effect).mean()
    specificity = torch.relu(0.02 - selected_effect + wrong_target_effect).mean()
    direction = torch.relu(-support).mean() + torch.relu(-counter).mean()
    return {
        "selected_control": selected_control,
        "specificity": specificity,
        "direction": direction,
        "total": selected_control + specificity + direction,
    }
