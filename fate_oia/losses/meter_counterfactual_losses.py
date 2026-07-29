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
    sign = target.unsqueeze(-1) * 2.0 - 1.0
    clean_score = (sign * clean_contributions).sum((-1, -2))
    corrupt_score = (sign * corrupt_contributions).sum((-1, -2))
    return torch.relu(margin + corrupt_score - clean_score).mean()


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
