from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_MIRROR_PAIRS = ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19))
DEFAULT_ACTION_MIRROR_PAIRS = ((2, 3),)
DEFAULT_GROUNDING_WEIGHTS = {
    "anchor": 0.10,
    "state": 0.10,
    "null": 0.03,
    "observability": 0.03,
    "discrimination": 0.05,
    "mirror": 0.05,
}


def _swap_factor_rows(value: Tensor, pairs: tuple[tuple[int, int], ...]) -> Tensor:
    result = value.clone()
    for left, right in pairs:
        result[:, left], result[:, right] = value[:, right].clone(), value[:, left].clone()
    return result


def mirror_equivariance_components(
    original: dict[str, Tensor],
    mirrored: dict[str, Tensor],
    *,
    factor_pairs: tuple[tuple[int, int], ...] = DEFAULT_MIRROR_PAIRS,
    action_pairs: tuple[tuple[int, int], ...] = DEFAULT_ACTION_MIRROR_PAIRS,
) -> dict[str, Tensor]:
    """Return independently auditable mirror objectives by prediction branch."""
    indices = sorted({index for pair in factor_pairs for index in pair})
    mirrored_anchor = torch.flip(
        _swap_factor_rows(mirrored["factor_anchor_map"], factor_pairs), dims=[-1]
    )
    mirrored_state = _swap_factor_rows(mirrored["factor_state_prob"], factor_pairs)
    mirrored_action = mirrored["action_logits_final"].clone()
    for left, right in action_pairs:
        mirrored_action[:, left], mirrored_action[:, right] = (
            mirrored["action_logits_final"][:, right].clone(),
            mirrored["action_logits_final"][:, left].clone(),
        )
    mirrored_reason = _swap_factor_rows(
        mirrored["reason_logits_final"].unsqueeze(-1), factor_pairs
    ).squeeze(-1)
    anchor_l1 = (
        original["factor_anchor_map"][:, indices]
        - mirrored_anchor[:, indices]
    ).abs().mean()
    state_l1 = (
        original["factor_state_prob"][:, indices]
        - mirrored_state[:, indices]
    ).abs().mean()
    action_l1 = (original["action_logits_final"] - mirrored_action).abs().mean()
    reason_l1 = (
        original["reason_logits_final"][:, indices]
        - mirrored_reason[:, indices]
    ).abs().mean()
    return {
        "anchor": anchor_l1,
        "state": state_l1,
        "action": action_l1,
        "reason": reason_l1,
    }


def mirror_equivariance_loss(
    original: dict[str, Tensor],
    mirrored: dict[str, Tensor],
    *,
    factor_pairs: tuple[tuple[int, int], ...] = DEFAULT_MIRROR_PAIRS,
    action_pairs: tuple[tuple[int, int], ...] = DEFAULT_ACTION_MIRROR_PAIRS,
) -> tuple[Tensor, dict[str, Any]]:
    """Compare paired original/mirror forwards, never same-image left/right states."""
    components = mirror_equivariance_components(
        original,
        mirrored,
        factor_pairs=factor_pairs,
        action_pairs=action_pairs,
    )
    per_factor_margin: dict[str, float] = {}
    for left, right in factor_pairs:
        for factor, partner in ((left, right), (right, left)):
            correct = (
                original["factor_anchor_map"][:, factor]
                - torch.flip(mirrored["factor_anchor_map"][:, partner], dims=[-1])
            ).abs().mean()
            wrong = (
                original["factor_anchor_map"][:, factor]
                - torch.flip(mirrored["factor_anchor_map"][:, factor], dims=[-1])
            ).abs().mean()
            per_factor_margin[str(factor)] = float((wrong - correct).detach())
    loss = sum(components.values())
    return loss, {
        "paired_forward": True,
        "anchor_l1": float(components["anchor"].detach()),
        "state_l1": float(components["state"].detach()),
        "action_l1": float(components["action"].detach()),
        "reason_l1": float(components["reason"].detach()),
        "per_factor_margin": per_factor_margin,
    }


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    weight = weight.to(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def source_weighted_anchor_loss(
    predicted: Tensor,
    target: Tensor,
    valid: Tensor,
    source_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    target = target.flatten(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-6)
    nll = -(target * predicted.clamp_min(1e-8).log()).sum(-1)
    intersection = (target * predicted).sum(-1)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (
        target.sum(-1) + predicted.sum(-1) + 1e-6
    )
    weight = valid.to(predicted) * source_weight.to(predicted)
    return _weighted_mean(nll, weight), _weighted_mean(dice, weight)


def conditional_state_ce(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    source_weight: Tensor,
) -> Tensor:
    safe_target = target.clamp_min(0)
    per = F.cross_entropy(logits.transpose(1, 2), safe_target, reduction="none")
    return _weighted_mean(per, valid.to(logits) * source_weight.to(logits))


def null_partition_calibration_loss(
    null_mass: Tensor,
    present_valid: Tensor,
    absent_valid: Tensor,
    source_weight: Tensor,
) -> Tensor:
    """Teach null only for reliable present/absent observations, never unknowns."""
    present = present_valid.to(torch.bool)
    absent = absent_valid.to(torch.bool)
    if bool((present & absent).any()):
        raise ValueError("A factor cannot be reliably present and absent together")
    valid = present | absent
    target = absent.to(null_mass)
    per = F.binary_cross_entropy(
        null_mass.clamp(1e-6, 1.0 - 1e-6), target, reduction="none"
    )
    return _weighted_mean(per, valid.to(null_mass) * source_weight.to(null_mass))


def _grounding_weights(weights: dict[str, float] | None) -> dict[str, float]:
    supplied = {} if weights is None else weights
    resolved = {
        key: float(supplied.get(key, default))
        for key, default in DEFAULT_GROUNDING_WEIGHTS.items()
    }
    # Older configs used one discrimination coefficient for both terms.
    if "mirror" not in supplied:
        resolved["mirror"] = resolved["discrimination"]
    return resolved


def observability_objective(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    source_weight: Tensor,
    tau: Tensor,
) -> tuple[Tensor, Tensor]:
    per = F.binary_cross_entropy_with_logits(logits, target.to(logits), reduction="none")
    weight = valid.to(logits) * source_weight.to(logits)
    bce = _weighted_mean(per, weight)
    probability = torch.sigmoid(logits)
    per_factor_weight = weight.sum(0)
    observed_mean = (probability * weight).sum(0) / per_factor_weight.clamp_min(1.0)
    available = per_factor_weight > 0
    coverage = (observed_mean - tau.to(logits)).abs()
    return bce, _weighted_mean(coverage, available.to(logits))


def discrimination_and_mirror_loss(
    output: dict[str, Tensor],
    targets: dict[str, Tensor],
    mirror_pairs: tuple[tuple[int, int], ...] = DEFAULT_MIRROR_PAIRS,
    mirrored_output: dict[str, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    token = output["factor_typed_token"]
    state = output["factor_state_prob"]
    source = targets["factor_source_weight"].to(token)
    predicted = output["factor_anchor_map"]
    target = targets["factor_anchor_map"].to(predicted).flatten(2)
    valid = targets["factor_anchor_valid"].to(predicted)
    same_type_terms: list[Tensor] = []
    background_terms: list[Tensor] = []
    for left, right in mirror_pairs:
        for correct, wrong_factor in ((left, right), (right, left)):
            correct_score = (predicted[:, correct] * target[:, correct]).sum(-1)
            wrong_score = (predicted[:, wrong_factor] * target[:, correct]).sum(-1)
            weight = source[:, correct] * valid[:, correct]
            same_type_terms.append(
                _weighted_mean(
                    torch.relu(0.05 + wrong_score - correct_score), weight
                )
            )
    for factor in range(predicted.shape[1]):
        correct_score = (predicted[:, factor] * target[:, factor]).sum(-1)
        background = 1.0 - target[:, factor].clamp(0, 1)
        background_score = (
            predicted[:, factor] * background
        ).sum(-1) / background.sum(-1).clamp_min(1.0)
        weight = source[:, factor] * valid[:, factor]
        background_terms.append(
            _weighted_mean(
                torch.relu(0.02 + background_score - correct_score), weight
            )
        )
    discrimination = (
        torch.cat(
            [
                torch.stack(same_type_terms),
                torch.stack(background_terms),
            ]
        ).mean()
        if same_type_terms and background_terms
        else token.new_zeros(())
    )
    mirror = (
        mirror_equivariance_loss(
            {
                key: value[: mirrored_output["action_logits_final"].shape[0]]
                for key, value in output.items()
                if key
                in {
                    "factor_anchor_map",
                    "factor_state_prob",
                    "action_logits_final",
                    "reason_logits_final",
                }
            },
            mirrored_output,
            factor_pairs=mirror_pairs,
        )[0]
        if mirrored_output is not None
        else token.new_zeros(())
    )
    return discrimination, mirror


def meter_grounding_loss(
    output: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    observability_tau: Tensor | None = None,
    mirrored_output: dict[str, Tensor] | None = None,
    mirror_pairs: tuple[tuple[int, int], ...] = DEFAULT_MIRROR_PAIRS,
    weights: dict[str, float] | None = None,
) -> dict[str, Tensor]:
    source = targets["factor_source_weight"].to(output["factor_anchor_map"])
    anchor_nll, anchor_dice = source_weighted_anchor_loss(
        output["factor_anchor_map"],
        targets["factor_anchor_map"],
        targets["factor_anchor_valid"],
        source,
    )
    state = conditional_state_ce(
        output["factor_state_logits"],
        targets["factor_state_target"],
        targets["factor_state_valid"],
        source,
    )
    # Derive training masks from the mirrored state contract. The explicit
    # present/absent fields are retained for audit, but are not a gradient
    # source because legacy mirror collation does not yet swap new fields.
    present_valid = targets["factor_state_valid"] & targets["factor_state_target"].eq(0)
    absent_valid = targets["factor_state_valid"] & targets["factor_state_target"].eq(1)
    null = null_partition_calibration_loss(
        output["factor_null_mass"], present_valid, absent_valid, source
    )
    tau = (
        torch.full((output["factor_observability"].shape[1],), 0.5, device=source.device)
        if observability_tau is None
        else observability_tau.to(source)
    )
    obs_bce, obs_coverage = observability_objective(
        output["factor_observability_logit"],
        targets["factor_observability"],
        targets["factor_observability_valid"],
        source,
        tau,
    )
    discrimination, mirror = discrimination_and_mirror_loss(
        output,
        targets,
        mirror_pairs=mirror_pairs,
        mirrored_output=mirrored_output,
    )
    anchor = anchor_nll + anchor_dice
    observability = obs_bce + obs_coverage
    resolved = _grounding_weights(weights)
    components = {
        "anchor": anchor,
        "state": state,
        "null": null,
        "observability": observability,
        "discrimination": discrimination,
        "mirror": mirror,
    }
    total = sum(resolved[name] * components[name] for name in resolved)
    return {
        "anchor_nll": anchor_nll,
        "anchor_dice": anchor_dice,
        "anchor": anchor,
        "state": state,
        "null": null,
        "observability_bce": obs_bce,
        "observability_coverage": obs_coverage,
        "observability": observability,
        "discrimination": discrimination,
        "mirror": mirror,
        "total": total,
    }
