from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _require_shape(value: Tensor, ndim: int, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    return value


def _expand_eligibility(
    value: Tensor | None,
    *,
    batch: int,
    factors: int,
    reference: Tensor,
) -> Tensor:
    if value is None:
        raise ValueError(
            "named eligibility is required; pass named_eligibility, "
            "predicate_named_mask, or predicate_groundable_mask"
        )
    value = torch.as_tensor(value, device=reference.device).float()
    if value.ndim == 1 and tuple(value.shape) == (factors,):
        value = value.view(1, factors).expand(batch, -1)
    elif tuple(value.shape) != (batch, factors):
        raise ValueError(
            "named_eligibility must have shape [F] or [B,F], "
            f"got {tuple(value.shape)}"
        )
    return value.clamp(0.0, 1.0)


def _expand_unnamed_prior(
    value: Tensor | None,
    *,
    batch: int,
    actions: int,
    patches: int,
    reference: Tensor,
    null_weight: Tensor | None,
) -> Tensor:
    if value is None:
        if null_weight is None:
            prior = reference.new_ones((batch, actions, patches), dtype=torch.float32)
        else:
            prior = null_weight.float().unsqueeze(-1).expand(-1, -1, patches)
    else:
        prior = torch.as_tensor(value, device=reference.device).float()
        if prior.ndim == 1 and prior.shape[0] == patches:
            prior = prior.view(1, 1, patches).expand(batch, actions, -1)
        elif prior.ndim == 2 and tuple(prior.shape) == (batch, patches):
            prior = prior.unsqueeze(1).expand(-1, actions, -1)
        elif tuple(prior.shape) != (batch, actions, patches):
            raise ValueError(
                "unnamed_prior must have shape [N], [B,N], or [B,A,N], "
                f"got {tuple(prior.shape)}"
            )
    # The epsilon is part of the unnamed numerator, so normalized masses still
    # sum to one exactly even when every named map is null.
    return prior.clamp_min(0.0) + 1e-8


def compute_named_unnamed_contributions(
    raw_contribution: Tensor,
    candidate_weight: Tensor,
    predicate_map: Tensor,
    *,
    named_eligibility: Tensor | None = None,
    predicate_groundable_mask: Tensor | None = None,
    predicate_named_mask: Tensor | None = None,
    unnamed_prior: Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Allocate signed patch evidence to named factors and an unnamed bypass.

    The calculation is intentionally streaming over the feature dimension:
    factor maps and candidate weights are combined with patch contributions,
    never with factor-specific patch feature tokens.  All reductions happen in
    float32 so the conservation contract remains tight under bf16 inputs.
    """
    raw_contribution = _require_shape(raw_contribution, 3, "raw_contribution")
    candidate_weight = _require_shape(candidate_weight, 3, "candidate_weight")
    predicate_map = _require_shape(predicate_map, 3, "predicate_map")
    batch, actions, patches = raw_contribution.shape
    if predicate_map.shape[0] != batch or predicate_map.shape[2] != patches:
        raise ValueError("predicate_map must match raw_contribution batch and patch dimensions")
    factors = predicate_map.shape[1]
    if candidate_weight.shape[:2] != (batch, actions):
        raise ValueError("candidate_weight must match raw_contribution batch and action dimensions")
    if candidate_weight.shape[-1] not in (factors, factors + 1):
        raise ValueError(
            "candidate_weight must have F real candidates or F+1 candidates "
            f"including null, got {candidate_weight.shape[-1]} for F={factors}"
        )

    raw = raw_contribution.float()
    candidates = candidate_weight.float()
    maps = predicate_map.float().clamp_min(0.0)
    null_weight = None
    if candidates.shape[-1] == factors + 1:
        null_weight = candidates[..., -1]
        candidates = candidates[..., :factors]

    eligibility = named_eligibility
    if eligibility is None:
        eligibility = predicate_named_mask
    if eligibility is None:
        eligibility = predicate_groundable_mask
    eligibility = _expand_eligibility(
        eligibility,
        batch=batch,
        factors=factors,
        reference=raw,
    )

    named_mass = (
        candidates.clamp_min(0.0).unsqueeze(-1)
        * eligibility.unsqueeze(1).unsqueeze(-1)
        * maps.unsqueeze(1)
    )
    unnamed_mass = _expand_unnamed_prior(
        unnamed_prior,
        batch=batch,
        actions=actions,
        patches=patches,
        reference=raw,
        null_weight=null_weight,
    )
    denominator = named_mass.sum(dim=2) + unnamed_mass
    denominator = denominator.clamp_min(float(eps))
    named_responsibility = named_mass / denominator.unsqueeze(2)
    unnamed_responsibility = unnamed_mass / denominator

    named_contribution = (
        named_responsibility * raw.unsqueeze(2)
    ).sum(dim=-1)
    unnamed_contribution = (unnamed_responsibility * raw).sum(dim=-1)
    raw_sum = raw.sum(dim=-1)
    conservation_error = named_contribution.sum(dim=-1) + unnamed_contribution - raw_sum
    responsibility_sum = named_responsibility.sum(dim=2) + unnamed_responsibility

    named_abs = named_contribution.abs().sum(dim=-1)
    unnamed_abs = unnamed_contribution.abs()
    fraction = named_abs / (named_abs + unnamed_abs).clamp_min(float(eps))
    effective_count = named_abs.square() / named_contribution.abs().square().sum(
        dim=-1
    ).clamp_min(float(eps))

    return {
        "action_named_responsibility": named_responsibility,
        "action_unnamed_responsibility": unnamed_responsibility,
        "action_responsibility_sum": responsibility_sum,
        "action_named_contribution": named_contribution,
        "action_unnamed_contribution": unnamed_contribution,
        "action_named_fraction": fraction,
        "action_effective_factor_count": effective_count,
        "action_conservation_error": conservation_error,
        "action_raw_contribution_sum": raw_sum,
    }


def compute_contribution_conservation(*args: Any, **kwargs: Any) -> dict[str, Tensor]:
    """Compatibility name for the SAVE named/unnamed conservation contract."""
    return compute_named_unnamed_contributions(*args, **kwargs)


def compute_named_unnamed_responsibility(
    *args: Any, **kwargs: Any
) -> dict[str, Tensor]:
    return compute_named_unnamed_contributions(*args, **kwargs)


def compute_named_unnamed_contribution(
    *args: Any, **kwargs: Any
) -> dict[str, Tensor]:
    return compute_named_unnamed_contributions(*args, **kwargs)


def named_unnamed_contribution_conservation(
    *args: Any, **kwargs: Any
) -> dict[str, Tensor]:
    return compute_named_unnamed_contributions(*args, **kwargs)


class SAVENamedUnnamedConservation(nn.Module):
    """Module wrapper for callers that keep conservation in a model graph."""

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        return compute_named_unnamed_contributions(*args, **kwargs)


def utility_teacher_target(
    control_margin: Tensor,
    selected_deletion_margin: Tensor,
    *,
    temperature: float = 0.10,
) -> Tensor:
    """Convert matched-control versus selected-deletion effects to a target."""
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    control = torch.as_tensor(control_margin).float()
    selected = torch.as_tensor(selected_deletion_margin, device=control.device).float()
    if control.shape != selected.shape:
        raise ValueError("control_margin and selected_deletion_margin must match")
    return torch.sigmoid((control - selected) / float(temperature))


compute_utility_teacher_target = utility_teacher_target


def signed_action_margin(action_logits: Tensor, action_targets: Tensor) -> Tensor:
    """Return ``(2*y-1) * z`` without changing the input graph."""
    if action_logits.shape != action_targets.shape:
        raise ValueError("action_logits and action_targets must have the same shape")
    return (action_targets.float() * 2.0 - 1.0) * action_logits.float()


def utility_counterfactual_loss(
    utility_logit: Tensor,
    target: Tensor | None = None,
    *,
    teacher_target: Tensor | None = None,
    sample_indices: Tensor | None = None,
    action_indices: Tensor | None = None,
    factor_indices: Tensor | None = None,
) -> Tensor:
    """Primary sparse teacher loss; the target is always detached."""
    if target is None:
        target = teacher_target
    if target is None:
        raise ValueError("utility_counterfactual_loss requires a teacher target")
    prediction = gather_sparse_utility_logits(
        utility_logit,
        sample_indices=sample_indices,
        action_indices=action_indices,
        factor_indices=factor_indices,
    )
    target = target.detach().to(device=prediction.device, dtype=torch.float32)
    if prediction.shape != target.shape:
        raise ValueError(
            "utility teacher target must match gathered utility predictions, "
            f"got {tuple(target.shape)} versus {tuple(prediction.shape)}"
        )
    return F.smooth_l1_loss(prediction.float(), target)


def gather_sparse_utility_logits(
    utility_logit: Tensor,
    *,
    sample_indices: Tensor | None = None,
    action_indices: Tensor | None = None,
    factor_indices: Tensor | None = None,
) -> Tensor:
    """Gather one utility prediction for every explicit teacher triple."""
    utility_logit = _require_shape(utility_logit, 3, "utility_logit")
    indices = (sample_indices, action_indices, factor_indices)
    if all(value is None for value in indices):
        return utility_logit
    if any(value is None for value in indices):
        raise ValueError("sample/action/factor indices must be provided together")
    sample = torch.as_tensor(sample_indices, device=utility_logit.device, dtype=torch.long)
    action = torch.as_tensor(action_indices, device=utility_logit.device, dtype=torch.long)
    factor = torch.as_tensor(factor_indices, device=utility_logit.device, dtype=torch.long)
    if sample.ndim != 1 or action.ndim != 1 or factor.ndim != 1:
        raise ValueError("sample/action/factor indices must be rank-1")
    if not (sample.shape == action.shape == factor.shape):
        raise ValueError("sample/action/factor indices must have identical shapes")
    if sample.numel() == 0:
        return utility_logit.new_empty((0,))
    if bool((sample < 0).any()) or bool((sample >= utility_logit.shape[0]).any()):
        raise ValueError("sample index is out of range")
    if bool((action < 0).any()) or bool((action >= utility_logit.shape[1]).any()):
        raise ValueError("action index is out of range")
    if bool((factor < 0).any()) or bool((factor >= utility_logit.shape[2]).any()):
        raise ValueError("factor index is out of range")
    return utility_logit[sample, action, factor]


utility_counterfactual_teacher_loss = utility_counterfactual_loss


def dense_utility_auxiliary_loss(
    utility_logit: Tensor,
    named_contribution: Tensor,
    action_targets: Tensor,
    *,
    weight: float = 0.02,
) -> Tensor:
    """Low-weight analytic coverage loss from named signed contribution."""
    if named_contribution.ndim != 3 or utility_logit.ndim != 3:
        raise ValueError("utility_logit and named_contribution must be [B,A,F]")
    if utility_logit.shape != named_contribution.shape:
        raise ValueError("utility_logit and named_contribution must have the same shape")
    if action_targets.shape != utility_logit.shape[:2]:
        raise ValueError("action_targets must have shape [B,A]")
    target = (action_targets.float().unsqueeze(-1) * 2.0 - 1.0) * named_contribution.float()
    return float(weight) * F.smooth_l1_loss(utility_logit.float(), target.detach())


utility_dense_auxiliary_loss = dense_utility_auxiliary_loss


def save_faithfulness_losses(
    output: dict[str, Tensor],
    action_targets: Tensor,
    *,
    utility_teacher_target_value: Tensor | None = None,
    teacher_target: Tensor | None = None,
    teacher_plan: Mapping[str, Any] | None = None,
    counterfactual_weight: float = 0.10,
    dense_weight: float = 0.02,
) -> dict[str, Tensor]:
    """Assemble the sparse teacher and dense auxiliary utility terms once."""
    if teacher_plan is None:
        value = output.get("teacher_plan")
        teacher_plan = value if isinstance(value, Mapping) else None
    if utility_teacher_target_value is None:
        utility_teacher_target_value = teacher_target
    if utility_teacher_target_value is None and teacher_plan is not None:
        utility_teacher_target_value = teacher_plan.get("utility_teacher_target")
    if utility_teacher_target_value is None:
        utility_teacher_target_value = output.get("utility_teacher_target")
    utility_logit = output["utility_logit"]
    teacher_loss = utility_logit.new_zeros(())
    teacher_prediction = utility_logit.new_empty((0,))
    teacher_target_output = utility_logit.new_empty((0,))
    if utility_teacher_target_value is not None:
        sample_indices = None if teacher_plan is None else teacher_plan.get("sample_indices")
        action_indices = None if teacher_plan is None else teacher_plan.get("action_indices")
        factor_indices = None if teacher_plan is None else teacher_plan.get("factor_indices")
        teacher_prediction = gather_sparse_utility_logits(
            utility_logit,
            sample_indices=sample_indices,
            action_indices=action_indices,
            factor_indices=factor_indices,
        )
        teacher_target_output = utility_teacher_target_value.detach().to(
            device=teacher_prediction.device,
            dtype=teacher_prediction.dtype,
        )
        teacher_loss = float(counterfactual_weight) * (
            utility_counterfactual_loss(
                utility_logit,
                teacher_target_output,
                sample_indices=sample_indices,
                action_indices=action_indices,
                factor_indices=factor_indices,
            )
        )
    elif teacher_plan is not None:
        raise RuntimeError("counterfactual teacher plan is missing utility targets")
    named = output.get("action_named_contribution")
    dense_loss = utility_logit.new_zeros(())
    dense_target = utility_logit.new_empty((0,))
    if named is not None:
        dense_target = (
            (action_targets.float().unsqueeze(-1) * 2.0 - 1.0)
            * named.detach().float()
        ).to(utility_logit)
        dense_loss = dense_utility_auxiliary_loss(
            utility_logit,
            named,
            action_targets,
            weight=dense_weight,
        )
    return {
        "loss_utility_cf": teacher_loss,
        "loss_utility_dense": dense_loss,
        "loss_utility": teacher_loss + dense_loss,
        "total": teacher_loss + dense_loss,
        "utility_teacher_prediction": teacher_prediction,
        "utility_teacher_target": teacher_target_output,
        "utility_dense_target": dense_target,
    }


save_faithfulness_loss = save_faithfulness_losses
named_unnamed_responsibility = compute_named_unnamed_contributions


__all__ = [
    "SAVENamedUnnamedConservation",
    "compute_contribution_conservation",
    "compute_named_unnamed_contributions",
    "compute_named_unnamed_contribution",
    "compute_named_unnamed_responsibility",
    "compute_utility_teacher_target",
    "dense_utility_auxiliary_loss",
    "gather_sparse_utility_logits",
    "named_unnamed_contribution_conservation",
    "named_unnamed_responsibility",
    "save_faithfulness_losses",
    "save_faithfulness_loss",
    "signed_action_margin",
    "utility_counterfactual_loss",
    "utility_counterfactual_teacher_loss",
    "utility_dense_auxiliary_loss",
    "utility_teacher_target",
]
