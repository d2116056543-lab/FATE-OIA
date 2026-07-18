from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from fate_oia.losses.mosaic_icdor_factor_losses import factor_positive_anchor_loss


_ANCHOR_SPLITS = frozenset(("train_core", "train_audit"))
_UNKNOWN = 0
_GEOMETRY_POSITIVE = 1
_REASON_ANCHOR_POSITIVE = 2
_GEOMETRY_AND_REASON_POSITIVE = 3
_WEAK_NEGATIVE = 4


def _evidence_mask(observations: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    value = observations.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"factor observations require a [B,F] {key}")
    return value


def _anchor_matrix(factors: Sequence[Mapping[str, Any]], reason_count: int, device: torch.device) -> torch.Tensor:
    matrix = torch.zeros(len(factors), reason_count, dtype=torch.bool, device=device)
    for factor_index, factor in enumerate(factors):
        if not isinstance(factor, Mapping):
            raise ValueError("factor definitions must be mappings")
        if "positive_reason_anchors" in factor and "reason_positive_anchors" in factor:
            if factor["positive_reason_anchors"] != factor["reason_positive_anchors"]:
                raise ValueError("conflicting positive reason anchor fields")
        anchors = factor.get("positive_reason_anchors", factor.get("reason_positive_anchors", ()))
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            raise ValueError("positive reason anchors must be a sequence")
        for reason_index in anchors:
            if type(reason_index) is not int or not 0 <= reason_index < reason_count:
                raise ValueError("positive reason anchors must be valid reason indices")
            matrix[factor_index, reason_index] = True
    return matrix


def build_factor_supervision(
    observations: Mapping[str, torch.Tensor],
    reason_targets: torch.Tensor | None,
    factors: Sequence[Mapping[str, Any]],
    *,
    split: str,
    allow_reason_anchors: bool = True,
) -> dict[str, torch.Tensor]:
    """Compose geometry, optional observed-positive anchors, and weak negatives.

    ``allow_reason_anchors`` exists only for legacy experiments. CREDO main
    training sets it to false: visual-factor measurement then consumes only
    image/geometry observations, while reasons remain target-side supervision.
    A zero reason label always remains unknown and cannot create a factor
    negative. Geometry wins whenever it co-occurs with an enabled anchor.
    """
    presence_target = _evidence_mask(observations, "presence_target")
    presence_known_mask = _evidence_mask(observations, "presence_known_mask")
    geometry_known_mask = _evidence_mask(observations, "geometry_known_mask")
    weak_negative_mask = _evidence_mask(observations, "weak_negative_mask")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a non-empty string")
    if not isinstance(allow_reason_anchors, bool):
        raise ValueError("allow_reason_anchors must be a boolean")
    if len(factors) != presence_target.shape[1]:
        raise ValueError("factor supervision batch or factor dimensions do not match")
    if allow_reason_anchors:
        if not isinstance(reason_targets, torch.Tensor) or reason_targets.ndim != 2:
            raise ValueError("reason targets must be [B,R] when reason anchors are enabled")
        if reason_targets.shape[0] != presence_target.shape[0]:
            raise ValueError("factor supervision batch or factor dimensions do not match")
    if any(value.shape != presence_target.shape for value in (presence_known_mask, geometry_known_mask, weak_negative_mask)):
        raise ValueError("factor observation tensors must share [B,F] shape")

    geometry_positive_mask = (geometry_known_mask > 0) & (presence_target > 0)
    if allow_reason_anchors and split in _ANCHOR_SPLITS:
        assert isinstance(reason_targets, torch.Tensor)
        anchors = _anchor_matrix(factors, reason_targets.shape[1], reason_targets.device)
        positive_anchor_mask = ((reason_targets > 0)[:, None, :] & anchors[None, :, :]).any(dim=-1)
    else:
        positive_anchor_mask = torch.zeros_like(geometry_positive_mask)

    reliable_negative_mask = (
        (presence_known_mask > 0)
        & (presence_target <= 0)
        & ~geometry_positive_mask
        & ~positive_anchor_mask
    )
    weak_negative_mask = (
        (weak_negative_mask > 0)
        & ~geometry_positive_mask
        & ~positive_anchor_mask
        & ~reliable_negative_mask
    )
    unknown_mask = ~(geometry_positive_mask | positive_anchor_mask | reliable_negative_mask | weak_negative_mask)
    supervision_mask = ~unknown_mask
    supervision_target = (geometry_positive_mask | positive_anchor_mask).to(dtype=presence_target.dtype)
    positive_anchor_weight = torch.zeros_like(presence_target, dtype=torch.float64)
    positive_anchor_weight[reliable_negative_mask] = 1.0
    positive_anchor_weight[weak_negative_mask] = 0.05
    positive_anchor_weight[positive_anchor_mask] = 0.35
    positive_anchor_weight[geometry_positive_mask] = 1.0

    supervision_code = torch.full_like(presence_target, _UNKNOWN, dtype=torch.long)
    supervision_code[reliable_negative_mask] = _WEAK_NEGATIVE
    supervision_code[weak_negative_mask] = _WEAK_NEGATIVE
    supervision_code[positive_anchor_mask] = _REASON_ANCHOR_POSITIVE
    supervision_code[geometry_positive_mask] = _GEOMETRY_POSITIVE
    supervision_code[geometry_positive_mask & positive_anchor_mask] = _GEOMETRY_AND_REASON_POSITIVE
    return {
        "supervision_target": supervision_target,
        "supervision_mask": supervision_mask,
        "geometry_positive_mask": geometry_positive_mask,
        "positive_anchor_mask": positive_anchor_mask,
        "reliable_negative_mask": reliable_negative_mask,
        "weak_negative_mask": weak_negative_mask,
        "unknown_mask": unknown_mask,
        "positive_anchor_weight": positive_anchor_weight,
        "supervision_source_code": supervision_code,
        "supervision_code": supervision_code,
    }
