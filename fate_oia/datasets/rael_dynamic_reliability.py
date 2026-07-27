"""Pure post-Hungarian reliability construction for RAEL training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from fate_oia.datasets.bdd100k_task_aware_index import RAELGroundingRecord
from fate_oia.datasets.rael_grounding_targets import DynamicGroundingBatch


@dataclass(frozen=True)
class DynamicReliabilityResult:
    q_ground: Tensor
    q_view: Tensor
    q_state: Tensor
    rho: Tensor
    q_view_sector: Tensor
    rho_clear: Tensor
    q_view_source: Tensor
    q_view_bootstrap_count: int
    rho_nonzero_rate: float
    ema_state: dict[str, Any]
    source_ids: tuple[tuple[str | None, ...], ...]


def _object_id(detection: Mapping[str, Any]) -> str | None:
    for key in ("id", "object_id", "track_id", "uuid"):
        value = detection.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _mask_iou(left: Tensor, right: Tensor) -> float:
    left_mask = left.detach().float() > 0.5
    right_mask = right.detach().float() > 0.5
    union = (left_mask | right_mask).sum()
    if int(union) == 0:
        return 0.0
    return float((left_mask & right_mask).sum() / union)


def _probability_consistency(left: Tensor, right: Tensor) -> float:
    value = (left.detach().float() * right.detach().float()).sum()
    return float(value.clamp(0.0, 1.0))


def _ema_update(values: dict[str, float], key: str, observation: float, beta: float) -> float:
    observation = max(0.0, min(1.0, float(observation)))
    current = values.get(key)
    updated = observation if current is None else beta * float(current) + (1.0 - beta) * observation
    values[key] = float(updated)
    return float(updated)


def build_dynamic_reliability(
    outputs: Mapping[str, Any],
    dynamic: DynamicGroundingBatch,
    records: Sequence[RAELGroundingRecord],
    road_valid: Mapping[str, Tensor],
    *,
    mirror_pairs: Tensor,
    sample_ids: Sequence[str],
    ema_state: Mapping[str, Any] | None,
    beta: float = 0.95,
) -> DynamicReliabilityResult:
    """Combine current matching, object-aligned mirror evidence, and state confidence."""

    masks = outputs.get("slot_masks")
    observability = outputs.get("slot_observability")
    type_probs = outputs.get("slot_type_probs")
    state_probs = outputs.get("slot_state_probs")
    if (
        not isinstance(masks, Tensor)
        or not isinstance(observability, Tensor)
        or not isinstance(type_probs, Tensor)
        or not isinstance(state_probs, Tensor)
        or masks.ndim != 4
        or observability.shape != masks.shape[:2]
        or type_probs.ndim != 3
        or state_probs.ndim != 3
        or type_probs.shape[:2] != state_probs.shape[:2]
    ):
        raise ValueError("dynamic reliability requires current masks/observability/type/state outputs")
    batch, slots = observability.shape
    entity_slots = type_probs.shape[1]
    if (
        len(dynamic.entity) != batch
        or len(records) != batch
        or len(sample_ids) != batch
        or entity_slots > slots
    ):
        raise ValueError("dynamic reliability batch dimensions are inconsistent")
    if not 0.0 <= beta < 1.0:
        raise ValueError("EMA beta must be in [0,1)")
    if (
        not isinstance(mirror_pairs, Tensor)
        or mirror_pairs.ndim != 2
        or mirror_pairs.shape[1] != 2
    ):
        raise ValueError("mirror_pairs must be [M,2]")

    device = observability.device
    dtype = observability.dtype
    q_ground = torch.zeros(batch, slots, device=device, dtype=dtype)
    q_view = torch.zeros_like(q_ground)
    q_view_source = torch.zeros(batch, slots, device=device, dtype=torch.long)
    q_state = torch.ones_like(q_ground)
    source_ids: list[list[str | None]] = [[None] * entity_slots for _ in range(batch)]
    assigned_slots: list[dict[str, int]] = []
    state = deepcopy(dict(ema_state or {}))
    object_ema = {str(key): float(value) for key, value in dict(state.get("objects", {})).items()}
    road_ema = {str(key): float(value) for key, value in dict(state.get("roads", {})).items()}

    for sample, (target, record) in enumerate(zip(dynamic.entity, records)):
        assignments = {item.slot_index: item for item in target.assignments}
        by_id: dict[str, int] = {}
        for slot, objectness in enumerate(target.objectness):
            assignment = assignments.get(slot)
            if assignment is not None:
                q_ground[sample, slot] = 1.0 / (1.0 + max(float(assignment.cost), 0.0))
                detection = record.detections[assignment.detection_index]
                identity = _object_id(detection)
                source_ids[sample][slot] = identity
                if identity is not None:
                    by_id[identity] = slot
                    if identity in object_ema:
                        q_view[sample, slot] = object_ema[identity]
                        q_view_source[sample, slot] = 1
            elif objectness.reliable:
                q_ground[sample, slot] = 1.0
        assigned_slots.append(by_id)

    type_confidence = type_probs.detach().max(dim=-1).values
    state_confidence = state_probs.detach().max(dim=-1).values
    traffic_probability = type_probs.detach()[..., 3]
    q_state[:, :entity_slots] = (
        type_confidence
        * ((1.0 - traffic_probability) + traffic_probability * state_confidence)
    ).to(device=device, dtype=dtype)

    drivable_valid = road_valid.get("drivable_valid_mask")
    boundary_valid = road_valid.get("boundary_valid_mask")
    if (
        not isinstance(drivable_valid, Tensor)
        or not isinstance(boundary_valid, Tensor)
        or drivable_valid.shape != (batch, 3)
        or boundary_valid.shape != (batch, 2)
        or slots < entity_slots + 5
    ):
        raise ValueError("road validity must contain bool [B,3] and [B,2]")
    q_ground[:, entity_slots : entity_slots + 5] = torch.cat(
        (drivable_valid, boundary_valid), dim=1
    ).to(device=device, dtype=dtype)
    q_ground[:, entity_slots + 5 :] = 1.0
    for sample, sample_id in enumerate(sample_ids):
        for road_index in range(5):
            key = f"{sample_id}:road:{road_index}"
            if key in road_ema:
                q_view[sample, entity_slots + road_index] = road_ema[key]
                q_view_source[sample, entity_slots + road_index] = 1

    for pair in mirror_pairs.detach().cpu().tolist():
        left, right = int(pair[0]), int(pair[1])
        if min(left, right) < 0 or max(left, right) >= batch:
            raise ValueError("mirror_pairs contain out-of-range indices")
        for identity in set(assigned_slots[left]).intersection(assigned_slots[right]):
            left_slot = assigned_slots[left][identity]
            right_slot = assigned_slots[right][identity]
            iou = _mask_iou(masks[left, left_slot].flip(-1), masks[right, right_slot])
            attribute = 0.5 * (
                _probability_consistency(type_probs[left, left_slot], type_probs[right, right_slot])
                + _probability_consistency(state_probs[left, left_slot], state_probs[right, right_slot])
            )
            score = _ema_update(object_ema, identity, 0.5 * (iou + attribute), beta)
            q_view[left, left_slot] = score
            q_view[right, right_slot] = score
            q_view_source[left, left_slot] = 2
            q_view_source[right, right_slot] = 2
        permutation = (2, 1, 0, 4, 3)
        for road_index, mirrored_index in enumerate(permutation):
            left_slot = entity_slots + road_index
            right_slot = entity_slots + mirrored_index
            score = _mask_iou(masks[left, left_slot].flip(-1), masks[right, right_slot])
            left_key = f"{sample_ids[left]}:road:{road_index}"
            right_key = f"{sample_ids[right]}:road:{mirrored_index}"
            left_score = _ema_update(road_ema, left_key, score, beta)
            right_score = _ema_update(road_ema, right_key, score, beta)
            q_view[left, left_slot] = left_score
            q_view[right, right_slot] = right_score
            q_view_source[left, left_slot] = 2
            q_view_source[right, right_slot] = 2

    feature_consistency = outputs.get("slot_feature_dropout_consistency")
    if feature_consistency is not None:
        if (
            not isinstance(feature_consistency, Tensor)
            or feature_consistency.shape != (batch, slots)
            or not bool(torch.isfinite(feature_consistency).all())
        ):
            raise ValueError(
                "slot_feature_dropout_consistency must be finite [B,20]"
            )
        feature_consistency = (
            feature_consistency.detach().to(device=device, dtype=dtype).clamp(0.0, 1.0)
        )
        bootstrap = (
            (q_view_source == 0)
            & (q_ground > 0.0)
            & (observability.detach().to(device=device) > 0.0)
            & (feature_consistency > 0.0)
        )
        q_view = torch.where(bootstrap, feature_consistency, q_view)
        q_view_source = torch.where(
            bootstrap, torch.full_like(q_view_source, 3), q_view_source
        )

    q_ground = q_ground.detach()
    q_view = q_view.detach()
    q_state = q_state.detach()
    rho = (
        observability.detach().to(dtype=dtype)
        * q_ground
        * q_view
        * q_state
    ).clamp(0.0, 1.0).detach()
    q_view_sector = q_view[:, entity_slots : entity_slots + 3].detach()
    road_outputs = outputs.get("grounding_outputs")
    road_outputs = road_outputs.get("road") if isinstance(road_outputs, Mapping) else None
    visibility = road_outputs.get("drivable_reliability") if isinstance(road_outputs, Mapping) else None
    if not isinstance(visibility, Tensor) or visibility.shape != (batch, 3):
        raise ValueError("dynamic reliability requires road sector visibility [B,3]")
    rho_clear = (visibility.detach().to(dtype=dtype) * q_view_sector).clamp(0.0, 1.0).detach()
    q_view_bootstrap_count = int((q_view_source == 3).sum().detach().cpu().item())
    rho_nonzero_rate = float((rho > 0.0).float().mean().detach().cpu().item())
    state["objects"] = object_ema
    state["roads"] = road_ema
    state["beta"] = float(beta)
    return DynamicReliabilityResult(
        q_ground=q_ground,
        q_view=q_view,
        q_state=q_state,
        rho=rho,
        q_view_sector=q_view_sector,
        rho_clear=rho_clear,
        q_view_source=q_view_source.detach(),
        q_view_bootstrap_count=q_view_bootstrap_count,
        rho_nonzero_rate=rho_nonzero_rate,
        ema_state=state,
        source_ids=tuple(tuple(row) for row in source_ids),
    )


__all__ = ["DynamicReliabilityResult", "build_dynamic_reliability"]
