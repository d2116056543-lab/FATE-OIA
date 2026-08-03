from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


BDD100K_SUPERVISION_SOURCE = "bdd100k"
DEFAULT_PREDICATE_GRID_SHAPE = (45, 80)
DEFAULT_SAVE_GROUNDING_WEIGHTS = {
    "anchor": 0.05,
    "state": 0.08,
    "null": 0.02,
    "matched_background": 0.03,
    "mirror": 0.02,
    "identity": 0.02,
}
DEFAULT_MIRROR_PAIRS = ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19))


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _get(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Missing SAVE grounding field; expected one of {names}")


def _optional(mapping: Mapping[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _factor_weight(
    value: Tensor,
    *,
    batch: int,
    factors: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    weight = value.to(device=device, dtype=dtype)
    if weight.ndim == 1:
        weight = weight.view(1, -1).expand(batch, -1)
    if tuple(weight.shape) != (batch, factors):
        raise ValueError("SAVE grounding factor weights must have shape [B,21]")
    return weight


def _predicate_masks(
    output: Mapping[str, Any], batch: int, factors: int, value: Tensor
) -> tuple[Tensor, Tensor]:
    groundable = _get(output, "predicate_groundable_mask").to(
        device=value.device, dtype=value.dtype
    )
    named = _get(output, "predicate_named_mask").to(
        device=value.device, dtype=value.dtype
    )
    if tuple(groundable.shape) != (factors,) or tuple(named.shape) != (factors,):
        raise ValueError("SAVE predicate masks must have shape [21]")
    return (
        groundable.view(1, -1).expand(batch, -1),
        named.view(1, -1).expand(batch, -1),
    )


def _source_and_provenance(
    targets: Mapping[str, Any],
    *,
    batch: int,
    factors: int,
    value: Tensor,
) -> tuple[Tensor, Tensor]:
    source = _factor_weight(
        _get(targets, "predicate_source_weight", "factor_source_weight"),
        batch=batch,
        factors=factors,
        device=value.device,
        dtype=value.dtype,
    )
    provenance = _factor_weight(
        _get(
            targets,
            "predicate_provenance_valid",
            "factor_provenance_valid",
        ),
        batch=batch,
        factors=factors,
        device=value.device,
        dtype=value.dtype,
    )
    return source, provenance


def predicate_anchor_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    predicted = _get(output, "predicate_map_raw", "predicate_map")
    if predicted.ndim != 3:
        raise ValueError("SAVE predicate_map must have shape [B,21,N]")
    batch, factors, patches = predicted.shape
    target = _get(targets, "predicate_anchor_map", "factor_anchor_map")
    target = target.to(device=predicted.device, dtype=predicted.dtype)
    if target.ndim > 3:
        target = target.flatten(2)
    if tuple(target.shape) != (batch, factors, patches):
        raise ValueError("SAVE anchor targets must match predicate_map")
    valid = _get(targets, "predicate_anchor_valid", "factor_anchor_valid").to(
        device=predicted.device, dtype=torch.bool
    )
    source, provenance = _source_and_provenance(
        targets,
        batch=batch,
        factors=factors,
        value=predicted,
    )
    groundable, _ = _predicate_masks(output, batch, factors, predicted)
    target = target.clamp_min(0.0)
    target_mass = target.sum(-1)
    valid_weight = (
        valid
        & (target_mass > 0)
        & (source > 0)
        & (provenance > 0)
        & (groundable > 0)
    ).to(predicted)
    normalized_target = target / target_mass.unsqueeze(-1).clamp_min(1e-6)
    nll = -(normalized_target * predicted.clamp_min(1e-8).log()).sum(-1)
    nll = nll / torch.log(
        predicted.new_tensor(float(max(patches, 2)))
    )
    intersection = (normalized_target * predicted).sum(-1)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (
        normalized_target.sum(-1) + predicted.sum(-1) + 1e-6
    )
    return (
        _weighted_mean(nll, valid_weight * source),
        _weighted_mean(dice, valid_weight * source),
    )


def predicate_state_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> Tensor:
    logits = _get(output, "predicate_state_logits")
    target = _get(targets, "predicate_state_target", "factor_state_target")
    valid = _get(targets, "predicate_state_valid", "factor_state_valid")
    if logits.ndim != 3 or target.ndim != 2 or valid.ndim != 2:
        raise ValueError("SAVE state tensors must have shapes [B,21,S], [B,21], [B,21]")
    batch, factors, states = logits.shape
    target = target.to(device=logits.device, dtype=torch.long)
    valid = valid.to(device=logits.device, dtype=torch.bool)
    source, provenance = _source_and_provenance(
        targets,
        batch=batch,
        factors=factors,
        value=logits,
    )
    groundable, _ = _predicate_masks(output, batch, factors, logits)
    state_valid_mask = _optional(
        output, "predicate_state_valid_mask", "factor_state_valid_mask"
    )
    target_valid = valid & (target >= 0) & (target < states)
    if state_valid_mask is not None:
        state_valid_mask = state_valid_mask.to(device=logits.device, dtype=torch.bool)
        if state_valid_mask.ndim == 2:
            state_valid_mask = state_valid_mask.unsqueeze(0).expand(
                batch, -1, -1
            )
        target_valid &= state_valid_mask.gather(
            -1, target.clamp(0, states - 1).unsqueeze(-1)
        ).squeeze(-1)
    # Latent predicates can be read as soft evidence, but BDD100K geometry
    # never supplies them with a grounding/state supervision path.
    target_valid &= (groundable > 0) & (provenance > 0) & (source > 0)
    safe_target = target.clamp(0, states - 1)
    per = F.cross_entropy(logits.transpose(1, 2), safe_target, reduction="none")
    return _weighted_mean(per, target_valid.to(logits) * source)


def predicate_null_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> Tensor:
    null_mass = _get(output, "predicate_null_mass")
    batch, factors = null_mass.shape
    present = _get(
        targets, "predicate_present_valid", "factor_present_valid"
    ).to(device=null_mass.device, dtype=torch.bool)
    absent = _get(
        targets, "predicate_absent_valid", "factor_absent_valid"
    ).to(device=null_mass.device, dtype=torch.bool)
    if bool((present & absent).any()):
        raise ValueError("A SAVE predicate cannot be both present and absent")
    source, provenance = _source_and_provenance(
        targets,
        batch=batch,
        factors=factors,
        value=null_mass,
    )
    groundable, _ = _predicate_masks(output, batch, factors, null_mass)
    valid = (
        (present | absent)
        & (groundable > 0)
        & (source > 0)
        & (provenance > 0)
    )
    target = absent.to(null_mass)
    with torch.autocast(device_type=null_mass.device.type, enabled=False):
        per = F.binary_cross_entropy(
            null_mass.float().clamp(1e-6, 1.0 - 1e-6),
            target.float(),
            reduction="none",
        )
    return _weighted_mean(per, valid.to(null_mass) * source)


def predicate_matched_background_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> Tensor:
    predicted = _get(output, "predicate_map_raw", "predicate_map")
    batch, factors, _ = predicted.shape
    target = _get(targets, "predicate_anchor_map", "factor_anchor_map").to(
        device=predicted.device, dtype=predicted.dtype
    )
    if target.ndim > 3:
        target = target.flatten(2)
    target_mass = target.sum(-1)
    target = target / target_mass.unsqueeze(-1).clamp_min(1e-6)
    valid = _get(targets, "predicate_anchor_valid", "factor_anchor_valid").to(
        device=predicted.device, dtype=torch.bool
    )
    source, provenance = _source_and_provenance(
        targets,
        batch=batch,
        factors=factors,
        value=predicted,
    )
    groundable, named = _predicate_masks(output, batch, factors, predicted)
    weight = valid.to(predicted) * (target_mass > 0).to(predicted)
    weight = weight * source * provenance * groundable * named
    correct = (predicted * target).sum(-1)
    background_mask = (target <= 0).to(predicted)
    background = (predicted * background_mask).sum(-1) / background_mask.sum(
        -1
    ).clamp_min(1.0)
    per = F.relu(0.02 + background - correct)
    return _weighted_mean(per, weight)


def _swap_rows(value: Tensor, pairs: tuple[tuple[int, int], ...]) -> Tensor:
    result = value.clone()
    for left, right in pairs:
        result[:, left], result[:, right] = value[:, right].clone(), value[:, left].clone()
    return result


def _validate_mirror_pairs(
    pairs: tuple[tuple[int, int], ...], factors: int
) -> None:
    used: set[int] = set()
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Each SAVE mirror pair must contain two indices")
        left, right = pair
        if (
            not isinstance(left, int)
            or not isinstance(right, int)
            or left == right
            or left < 0
            or right < 0
            or left >= factors
            or right >= factors
            or left in used
            or right in used
        ):
            raise ValueError(f"Invalid SAVE mirror pair: {pair}")
        used.update((left, right))


def _horizontal_flip_predicate_map(
    value: Tensor, grid_shape: tuple[int, int]
) -> Tensor:
    if (
        len(grid_shape) != 2
        or not all(isinstance(size, int) for size in grid_shape)
        or any(size <= 0 for size in grid_shape)
    ):
        raise ValueError("SAVE predicate grid_shape must contain positive H and W")
    height, width = grid_shape
    if value.shape[-1] != height * width:
        raise ValueError(
            "SAVE predicate map N must equal grid_shape H*W for mirror grounding"
        )
    batch, factors, _ = value.shape
    grid = value.reshape(batch, factors, height, width)
    return torch.flip(grid, dims=[-1]).reshape_as(value)


def predicate_mirror_loss(
    output: Mapping[str, Any],
    mirrored_output: Mapping[str, Any],
    *,
    mirror_pairs: tuple[tuple[int, int], ...] = DEFAULT_MIRROR_PAIRS,
    valid: Tensor | None = None,
    grid_shape: tuple[int, int] = DEFAULT_PREDICATE_GRID_SHAPE,
) -> Tensor:
    original_map = _get(output, "predicate_map_raw", "predicate_map")
    candidate_map = _get(mirrored_output, "predicate_map_raw", "predicate_map")
    if original_map.ndim != 3 or original_map.shape != candidate_map.shape:
        raise ValueError("SAVE mirrored predicate maps must have equal [B,21,N] shapes")
    batch, factors, _ = original_map.shape
    _validate_mirror_pairs(mirror_pairs, factors)
    mirrored_map = _horizontal_flip_predicate_map(
        _swap_rows(candidate_map, mirror_pairs), grid_shape
    )
    original_state = _get(
        output, "predicate_state_prob_raw", "predicate_state_prob"
    )
    candidate_state = _get(
        mirrored_output, "predicate_state_prob_raw", "predicate_state_prob"
    )
    if original_state.ndim != 3 or original_state.shape != candidate_state.shape:
        raise ValueError("SAVE mirrored predicate states must have equal [B,21,S] shapes")
    mirrored_state = _swap_rows(candidate_state, mirror_pairs)
    indices = sorted({index for pair in mirror_pairs for index in pair})
    if not indices:
        return original_map.new_zeros(())
    pair_weight = original_map.new_zeros(batch, factors)
    pair_weight[:, indices] = 1.0
    if valid is not None:
        pair_weight = pair_weight * _factor_weight(
            valid,
            batch=batch,
            factors=factors,
            device=original_map.device,
            dtype=original_map.dtype,
        )
    per_factor = (
        original_map.sub(mirrored_map).abs().mean(-1)
        + original_state.sub(mirrored_state).abs().mean(-1)
    )
    return _weighted_mean(per_factor, pair_weight)


def predicate_identity_loss(output: Mapping[str, Any]) -> Tensor:
    query = _optional(output, "predicate_ontology_query")
    target = _optional(output, "predicate_ontology_target")
    state_query = _optional(output, "predicate_state_ontology_query")
    state_target = _optional(output, "predicate_state_ontology_target")
    if query is None or target is None or state_query is None or state_target is None:
        value = _get(output, "predicate_map_raw", "predicate_map")
        return value.new_zeros(())
    factor = 1.0 - F.cosine_similarity(query, target.detach(), dim=-1)
    state = 1.0 - F.cosine_similarity(
        state_query, state_target.detach(), dim=-1
    )
    return factor.mean() + state.mean()


def _resolved_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    supplied = {} if weights is None else weights
    return {
        name: float(supplied.get(name, default))
        for name, default in DEFAULT_SAVE_GROUNDING_WEIGHTS.items()
    }


def save_grounding_loss(
    output: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    split: str,
    supervision_source: str,
    mirrored_output: Mapping[str, Any] | None = None,
    mirror_pairs: tuple[tuple[int, int], ...] | None = None,
    mirror_grid_shape: tuple[int, int] = DEFAULT_PREDICATE_GRID_SHAPE,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Tensor]:
    """Compute train-only BDD100K weak grounding objectives.

    Targets are never accepted for evaluation splits.  Unknown targets are
    invalid rather than negative, and latent predicates are excluded from all
    geometry/state grounding terms while remaining available to action.
    """
    if str(supervision_source).strip().casefold() != BDD100K_SUPERVISION_SOURCE:
        raise ValueError("SAVE weak grounding supervision source must be BDD100K")
    if split != "train":
        raise ValueError("SAVE BDD100K weak grounding is train-only")
    anchor_nll, anchor_dice = predicate_anchor_loss(output, targets)
    state = predicate_state_loss(output, targets)
    null = predicate_null_loss(output, targets)
    matched_background = predicate_matched_background_loss(output, targets)
    if mirrored_output is None:
        mirror = _get(output, "predicate_map_raw", "predicate_map").new_zeros(())
    else:
        resolved_pairs = (
            tuple(mirror_pairs)
            if mirror_pairs is not None
            else tuple(_get(output, "predicate_mirror_pairs"))
        )
        predicted = _get(output, "predicate_map_raw", "predicate_map")
        batch, factors, _ = predicted.shape
        source, provenance = _source_and_provenance(
            targets,
            batch=batch,
            factors=factors,
            value=predicted,
        )
        groundable, _ = _predicate_masks(output, batch, factors, predicted)
        mirror_valid = source * provenance * groundable
        mirror = predicate_mirror_loss(
            output,
            mirrored_output,
            mirror_pairs=resolved_pairs,
            valid=mirror_valid,
            grid_shape=mirror_grid_shape,
        )
    identity = predicate_identity_loss(output)
    anchor = anchor_nll + anchor_dice
    resolved = _resolved_weights(weights)
    total = (
        resolved["anchor"] * anchor
        + resolved["state"] * state
        + resolved["null"] * null
        + resolved["matched_background"] * matched_background
        + resolved["mirror"] * mirror
        + resolved["identity"] * identity
    )
    return {
        "anchor_nll": anchor_nll,
        "anchor_dice": anchor_dice,
        "anchor": anchor,
        "state": state,
        "null": null,
        "observability": null,
        "matched_background": matched_background,
        "discrimination": matched_background,
        "mirror": mirror,
        "identity": identity,
        "ontology_identity": identity,
        "total": total,
    }


predicate_grounding_loss = save_grounding_loss


__all__ = [
    "BDD100K_SUPERVISION_SOURCE",
    "DEFAULT_MIRROR_PAIRS",
    "DEFAULT_PREDICATE_GRID_SHAPE",
    "DEFAULT_SAVE_GROUNDING_WEIGHTS",
    "predicate_anchor_loss",
    "predicate_grounding_loss",
    "predicate_identity_loss",
    "predicate_matched_background_loss",
    "predicate_mirror_loss",
    "predicate_null_loss",
    "predicate_state_loss",
    "save_grounding_loss",
]
