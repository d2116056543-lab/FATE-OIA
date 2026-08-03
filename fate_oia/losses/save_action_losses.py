from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


SAVE_ACTION_LOSS_WEIGHTS = {
    "action_final": 1.00,
    "action_base": 0.35,
    "action_evidence_aux": 0.20,
    "action_utility_cf": 0.10,
    "action_utility_dense": 0.02,
    "action_sufficiency": 0.08,
    "action_necessity": 0.08,
    "action_control": 0.04,
    "action_preserve": 0.02,
    "action_soft_f1": 0.03,
    "action_cardinality": 0.02,
    "action_easy": 0.03,
}


def asymmetric_multilabel_elements(
    logits: Tensor,
    target: Tensor,
    gamma_negative: float = 2.0,
) -> Tensor:
    """Return per-entry ASL terms without reducing the action dimension."""
    if logits.shape != target.shape:
        raise ValueError(f"logits and target must have the same shape, got {logits.shape} and {target.shape}")
    probability = torch.sigmoid(logits.float())
    target = target.to(probability)
    return -target * torch.log(probability.clamp_min(1e-6)) - (
        (1.0 - target)
        * probability.pow(float(gamma_negative))
        * torch.log((1.0 - probability).clamp_min(1e-6))
    )


def asymmetric_multilabel_loss(
    logits: Tensor,
    target: Tensor,
    gamma_negative: float = 2.0,
) -> Tensor:
    return asymmetric_multilabel_elements(logits, target, gamma_negative).mean()


def soft_f1_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.to(probability)
    tp = (probability * target).sum(0)
    fp = (probability * (1.0 - target)).sum(0)
    fn = ((1.0 - probability) * target).sum(0)
    return 1.0 - ((2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)).mean()


def dense_utility_loss(
    utility_logits: Tensor,
    action_named_contribution: Tensor,
    target: Tensor,
) -> Tensor:
    """Train dense utility only as a low-weight signed contribution auxiliary."""
    if utility_logits.shape != action_named_contribution.shape:
        raise ValueError("utility_logits and named contribution must have the same shape")
    signed_target = (2.0 * target.unsqueeze(-1).to(action_named_contribution) - 1.0) * action_named_contribution
    return F.smooth_l1_loss(utility_logits, signed_target.detach())


def counterfactual_utility_loss(
    utility_prediction: Tensor,
    teacher_target: Tensor,
) -> Tensor:
    if utility_prediction.shape != teacher_target.shape:
        raise ValueError("utility prediction and teacher target must have the same shape")
    return F.smooth_l1_loss(utility_prediction, teacher_target.detach())


def evidence_auxiliary_action_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Keep the evidence decoder trainable while the formal ramp is zero."""
    return asymmetric_multilabel_loss(logits, target)


def easy_sample_nonregression_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    target: Tensor,
    *,
    easy_margin: float = 0.5,
) -> Tensor:
    """Keep already-easy base decisions from paying a new regression cost."""
    final_elements = asymmetric_multilabel_elements(final_logits, target).mean(-1)
    base_elements = asymmetric_multilabel_elements(base_logits.detach(), target).mean(-1)
    signed_base_margin = (2.0 * target - 1.0) * base_logits.detach()
    eligible = signed_base_margin.mean(-1) > float(easy_margin)
    if not bool(eligible.any()):
        return final_logits.new_zeros(())
    return torch.relu(final_elements - base_elements)[eligible].mean()


def _first_tensor(output: Mapping[str, Any], *names: str) -> Tensor:
    for name in names:
        value = output.get(name)
        if isinstance(value, Tensor):
            return value
    raise KeyError(f"output is missing all of {names}")


def _optional_term(output: Mapping[str, Any], anchor: Tensor, *names: str) -> Tensor:
    for name in names:
        value = output.get(name)
        if value is not None:
            if not isinstance(value, Tensor):
                value = anchor.new_tensor(value)
            return value.float().mean()
    return anchor.new_zeros(())


def _weight(weights: Mapping[str, float], key: str) -> float:
    return float(weights.get(key, SAVE_ACTION_LOSS_WEIGHTS[key]))


def save_action_loss(
    output: Mapping[str, Any],
    target: Tensor,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Tensor]:
    """Build the single-owner SAVE action loss bundle.

    Optional utility and intervention values are read once from the output;
    absent mechanisms contribute an explicit zero rather than silently
    restoring a legacy V3 loss term.
    """
    weights = weights or {}
    final_logits = _first_tensor(output, "action_logits_final")
    base_logits = _first_tensor(output, "action_logits_base", "action_logits_visual")
    evidence_aux_logits = _first_tensor(
        output, "action_logits_evidence_aux", "action_logits_evidence_auxiliary"
    )
    if final_logits.shape != target.shape:
        raise ValueError("action target must match action logit shape")
    target = target.to(final_logits)

    terms = {
        "final": asymmetric_multilabel_loss(final_logits, target),
        "base": asymmetric_multilabel_loss(base_logits, target),
        "evidence_aux": evidence_auxiliary_action_loss(evidence_aux_logits, target),
        "utility_cf": _optional_term(
            output, final_logits, "utility_loss_cf", "action_utility_cf_loss"
        ),
        "utility_dense": _optional_term(
            output, final_logits, "utility_loss_dense", "action_utility_dense_loss"
        ),
        "sufficiency": _optional_term(
            output, final_logits, "action_sufficiency_loss", "sufficiency_loss"
        ),
        "necessity": _optional_term(
            output, final_logits, "action_necessity_loss", "necessity_loss"
        ),
        "control": _optional_term(
            output, final_logits, "action_control_loss", "control_loss"
        ),
        "preserve": _optional_term(
            output, final_logits, "action_preserve_loss", "preserve_loss"
        ),
        "soft_f1": soft_f1_loss(final_logits, target),
        "cardinality": F.smooth_l1_loss(
            torch.sigmoid(final_logits.float()).sum(-1), target.float().sum(-1)
        ),
        "easy": easy_sample_nonregression_loss(final_logits, base_logits, target),
    }
    weighted_terms = (
        ("final", "action_final"),
        ("base", "action_base"),
        ("evidence_aux", "action_evidence_aux"),
        ("utility_cf", "action_utility_cf"),
        ("utility_dense", "action_utility_dense"),
        ("sufficiency", "action_sufficiency"),
        ("necessity", "action_necessity"),
        ("control", "action_control"),
        ("preserve", "action_preserve"),
        ("soft_f1", "action_soft_f1"),
        ("cardinality", "action_cardinality"),
        ("easy", "action_easy"),
    )
    total = final_logits.new_zeros(())
    for term_name, weight_name in weighted_terms:
        total = total + _weight(weights, weight_name) * terms[term_name]
    return {**terms, "total": total}


def save_action_loss_per_sample(
    output: Mapping[str, Any],
    target: Tensor,
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    """Return the deploy/base/aux action terms per example for audit logs."""
    weights = weights or {}
    final_logits = _first_tensor(output, "action_logits_final")
    base_logits = _first_tensor(output, "action_logits_base", "action_logits_visual")
    evidence_aux_logits = _first_tensor(
        output, "action_logits_evidence_aux", "action_logits_evidence_auxiliary"
    )
    target = target.to(final_logits)
    return (
        _weight(weights, "action_final")
        * asymmetric_multilabel_elements(final_logits, target).mean(-1)
        + _weight(weights, "action_base")
        * asymmetric_multilabel_elements(base_logits, target).mean(-1)
        + _weight(weights, "action_evidence_aux")
        * asymmetric_multilabel_elements(evidence_aux_logits, target).mean(-1)
    )


save_action_losses = save_action_loss
build_save_action_loss = save_action_loss
save_action_loss_bundle = save_action_loss


__all__ = [
    "SAVE_ACTION_LOSS_WEIGHTS",
    "asymmetric_multilabel_elements",
    "asymmetric_multilabel_loss",
    "build_save_action_loss",
    "counterfactual_utility_loss",
    "dense_utility_loss",
    "easy_sample_nonregression_loss",
    "evidence_auxiliary_action_loss",
    "save_action_loss",
    "save_action_loss_bundle",
    "save_action_loss_per_sample",
    "save_action_losses",
    "soft_f1_loss",
]
