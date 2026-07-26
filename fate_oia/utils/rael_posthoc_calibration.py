"""P15 frozen, deterministic posthoc calibration for RAEL logits.

This file deliberately contains no model state, optimizer, gradient path, or
trainer/evaluator integration.  Fitting consumes one frozen ``train_calib``
snapshot and returns a JSON-serializable deployment result.  Applying that
result separates the unchanged raw ranking source from strict threshold
decisions and a float64 diagnostic margin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import re
from typing import Any

import torch
from torch import Tensor

from fate_oia.losses.rael_pu_losses import canonicalize_sample_id


SCHEMA_VERSION = "RAEL_P15_POSTHOC_CALIBRATION_V1"
CANDIDATE_SCHEMA_VERSION = "RAEL_P15_CALIBRATION_CANDIDATE_V1"
TARGET_COUNTS = frozenset((4, 21))
RMS_FRACTION = 0.35
MAX_RAW_MF1_DROP = 0.005
SHRINKAGE_STRENGTH = 20.0
SHRINKAGE_FORMULA_VERSION = "support_weighted_group_shrinkage_v1"
THRESHOLD_GRID = tuple(round(-2.0 + 0.1 * index, 6) for index in range(41))
TEMPERATURE_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
CANDIDATE_KINDS = (
    "global_threshold",
    "group_threshold",
    "shrinkage_per_label_threshold",
    "positive_temperature_threshold",
)
PRIMARY_CANDIDATE_KINDS = CANDIDATE_KINDS[1:]
DIGEST_FIELD = "payload_sha256"
INTEGRITY_MODEL = "internal_consistency+accidental_corruption"
SHA256_LIMITATION = "sha256_detects_accidental_corruption_only_not_adversarial_resigning"
_SAFE_GROUP_ID = re.compile(r"[A-Za-z0-9_.:-]+\Z")
_FAMILY_PROVENANCE = {
    "global_threshold": {"family": "global", "scope": "all_labels", "temperature_mode": "identity"},
    "group_threshold": {"family": "group", "scope": "typed_group", "temperature_mode": "identity"},
    "shrinkage_per_label_threshold": {
        "family": "shrinkage",
        "scope": "per_label_support_weighted_group",
        "temperature_mode": "identity",
    },
    "positive_temperature_threshold": {"family": "temperature", "scope": "per_label", "temperature_mode": "grid_search"},
}


def _require_cpu_float32(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{name} must be a CPU float32 tensor")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] not in TARGET_COUNTS:
        raise ValueError(f"{name} must be [B,4] or [B,21]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _require_binary_labels(labels: Tensor, *, shape: torch.Size) -> None:
    _require_cpu_float32("labels", labels)
    if labels.shape != shape:
        raise ValueError("labels must match raw_logits shape")
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("labels must be strictly binary 0/1")


def _source_descriptor(
    *,
    stable_ids: Sequence[str | int] | None,
    split_hash: str | None,
    batch: int,
) -> dict[str, Any]:
    if stable_ids is None and split_hash is None:
        raise ValueError("fit requires stable_ids or split_hash")
    if stable_ids is not None:
        if isinstance(stable_ids, (str, bytes, Mapping)):
            raise TypeError("stable_ids must be a sequence")
        values = list(stable_ids)
        if len(values) != batch:
            raise ValueError("stable_ids must have one value per calibration row")
        canonical_ids = [canonicalize_sample_id(value) for value in values]
        if len(set(canonical_ids)) != batch:
            raise ValueError("stable_ids must be unique after P12 canonicalization")
        payload = json.dumps(canonical_ids, ensure_ascii=True, separators=(",", ":"))
        resolved_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return {
            "split_hash": resolved_hash,
            "stable_id_count": batch,
            "provided_split_hash": split_hash,
            "canonicalization": "P12_canonicalize_sample_id",
            "canonical_id_order": "input_row_order",
        }
    if not isinstance(split_hash, str) or not split_hash:
        raise ValueError("split_hash must be a non-empty string")
    return {
        "split_hash": split_hash,
        "stable_id_count": 0,
        "provided_split_hash": split_hash,
        "canonicalization": "provided_split_hash_only",
        "canonical_id_order": "not_available",
    }


def _canonical_group_id(group_id: int | str) -> str:
    """Type-preserve group IDs so integer and string namespaces cannot collide."""

    if isinstance(group_id, bool):
        raise TypeError("group_ids must not contain bool")
    if isinstance(group_id, int):
        return f"int:{group_id}"
    if isinstance(group_id, str):
        if not group_id or group_id.strip() != group_id or not _SAFE_GROUP_ID.fullmatch(group_id):
            raise ValueError("group_ids strings must be non-empty safe identifiers without whitespace")
        return f"str:{group_id}"
    raise TypeError("group_ids must contain non-bool int or safe str values")


def _normalize_groups(group_ids: Sequence[int | str], *, targets: int) -> tuple[list[str], list[str]]:
    if isinstance(group_ids, (str, bytes, Mapping)):
        raise TypeError("group_ids must be an explicit per-label sequence")
    groups = list(group_ids)
    if len(groups) != targets:
        raise ValueError("group_ids must provide one group per label")
    canonical = [_canonical_group_id(group) for group in groups]
    return canonical, sorted(set(canonical))


def _f1_per_label(decisions: Tensor, labels: Tensor) -> Tensor:
    prediction = decisions.to(dtype=torch.float32)
    target = labels.to(dtype=torch.float32)
    true_positive = (prediction * target).sum(dim=0)
    false_positive = (prediction * (1.0 - target)).sum(dim=0)
    false_negative = ((1.0 - prediction) * target).sum(dim=0)
    denominator = 2.0 * true_positive + false_positive + false_negative
    return torch.where(denominator > 0.0, 2.0 * true_positive / denominator, torch.zeros_like(denominator))


def _average_precision(scores: Tensor, labels: Tensor) -> float:
    """Tie-grouped binary AP, deterministic under input-row reordering."""

    positives = int(labels.sum().item())
    if positives == 0:
        return 0.0
    ordered_scores, order = torch.sort(scores.to(dtype=torch.float64), descending=True, stable=True)
    ordered_labels = labels.index_select(0, order).to(dtype=torch.float64)
    change = torch.ones_like(ordered_scores, dtype=torch.bool)
    if ordered_scores.numel() > 1:
        change[1:] = ordered_scores[1:] != ordered_scores[:-1]
    starts = torch.nonzero(change, as_tuple=False).reshape(-1)
    ends = torch.cat((starts[1:], torch.tensor([ordered_scores.numel()], dtype=starts.dtype)))
    cumulative_positive = ordered_labels.cumsum(dim=0)
    contribution = 0.0
    previous_positive = 0.0
    for start, end in zip(starts.tolist(), ends.tolist()):
        positives_in_group = float(ordered_labels[start:end].sum().item())
        if positives_in_group:
            precision = float(cumulative_positive[end - 1].item()) / float(end)
            contribution += precision * positives_in_group
        previous_positive += positives_in_group
    return contribution / float(positives)


def multi_label_metrics(logits: Tensor, labels: Tensor) -> dict[str, Any]:
    """Metrics for already frozen logits; prediction cutoff is zero."""

    _require_cpu_float32("logits", logits)
    _require_binary_labels(labels, shape=logits.shape)
    per_label_f1 = _f1_per_label(logits > 0.0, labels)
    average_precision = [_average_precision(logits[:, index], labels[:, index]) for index in range(logits.shape[1])]
    return {
        "mf1": float(per_label_f1.mean().item()),
        "per_label_f1": [float(value) for value in per_label_f1.tolist()],
        "map": float(sum(average_precision) / len(average_precision)),
        "per_label_ap": average_precision,
    }


def _decision_from_raw(raw_logits: Tensor, threshold: Tensor, temperature: Tensor) -> Tensor:
    """Compute deploy decisions exactly without manufacturing ranking logits."""

    raw64 = raw_logits.detach().to(dtype=torch.float64)
    threshold64 = threshold.detach().to(device=raw_logits.device, dtype=torch.float64)
    temperature64 = temperature.detach().to(device=raw_logits.device, dtype=torch.float64)
    return raw64 > temperature64.unsqueeze(0) * threshold64.unsqueeze(0)


def _decision_ranking_metrics(raw_logits: Tensor, labels: Tensor, decisions: Tensor) -> dict[str, Any]:
    """F1 comes from deploy decisions; AP is always measured on raw ranking."""

    if decisions.shape != raw_logits.shape or decisions.dtype != torch.bool:
        raise ValueError("decisions must be boolean and match raw_logits")
    per_label_f1 = _f1_per_label(decisions, labels)
    average_precision = [_average_precision(raw_logits[:, index], labels[:, index]) for index in range(raw_logits.shape[1])]
    return {
        "mf1": float(per_label_f1.mean().item()),
        "per_label_f1": [float(value) for value in per_label_f1.tolist()],
        "map": float(sum(average_precision) / len(average_precision)),
        "per_label_ap": average_precision,
        "ranking_source": "raw_logits",
    }


def _decision_metric_stats(decisions: Tensor, labels: Tensor) -> dict[str, Any]:
    """Frozen per-label confusion counts used for posthoc semantic validation."""

    if decisions.shape != labels.shape or decisions.dtype != torch.bool:
        raise ValueError("decisions must be boolean and match labels")
    prediction = decisions.to(dtype=torch.int64)
    target = labels.to(dtype=torch.int64)
    true_positive = (prediction * target).sum(dim=0)
    false_positive = (prediction * (1 - target)).sum(dim=0)
    false_negative = ((1 - prediction) * target).sum(dim=0)
    true_negative = ((1 - prediction) * (1 - target)).sum(dim=0)
    return {
        "sample_count": int(labels.shape[0]),
        "true_positive": [int(value) for value in true_positive.tolist()],
        "false_positive": [int(value) for value in false_positive.tolist()],
        "false_negative": [int(value) for value in false_negative.tolist()],
        "true_negative": [int(value) for value in true_negative.tolist()],
        "support": [int(value) for value in target.sum(dim=0).tolist()],
    }


def _candidate_metrics(raw_logits: Tensor, labels: Tensor, threshold: Tensor, temperature: Tensor) -> dict[str, Any]:
    return _decision_ranking_metrics(raw_logits, labels, _decision_from_raw(raw_logits, threshold, temperature))


def _threshold_rms_is_compliant(threshold: Tensor, *, raw_logit_rms: float, fraction: float = RMS_FRACTION) -> bool:
    if not isinstance(raw_logit_rms, float) or not math.isfinite(raw_logit_rms) or raw_logit_rms < 0.0:
        raise ValueError("raw_logit_rms must be a finite nonnegative float")
    if not isinstance(fraction, float) or not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0,1)")
    rms = float(threshold.detach().to(dtype=torch.float64).square().mean().sqrt().item())
    bound = fraction * raw_logit_rms
    tolerance = 1.0e-7 * max(1.0, abs(bound))
    return rms < bound - tolerance


def _candidate_record(
    *,
    kind: str,
    raw_logits: Tensor,
    labels: Tensor,
    threshold: Tensor,
    temperature: Tensor,
    search: dict[str, Any],
    shrinkage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if kind not in CANDIDATE_KINDS:
        raise ValueError("candidate kind must be one of the four P15 families")
    if threshold.ndim != 1 or temperature.ndim != 1 or threshold.shape != temperature.shape:
        raise ValueError("candidate threshold and temperature must be matching [K] tensors")
    if not bool(torch.isfinite(threshold).all()) or not bool(torch.isfinite(temperature).all()):
        raise ValueError("candidate threshold and temperature must be finite")
    allowed_temperatures = {float(value) for value in TEMPERATURE_GRID}
    if any(float(value) not in allowed_temperatures for value in temperature.detach().cpu().tolist()):
        raise ValueError("candidate temperature must come from the declared discrete grid")
    if not isinstance(search, dict):
        raise TypeError("candidate search metadata must be a dictionary")
    resolved_search = copy.deepcopy(search)
    existing_temperature_grid = resolved_search.get("temperature_grid")
    if existing_temperature_grid is not None and existing_temperature_grid != list(TEMPERATURE_GRID):
        raise ValueError("candidate search temperature grid must match the P15 grid")
    resolved_search["temperature_grid"] = list(TEMPERATURE_GRID)
    raw_rms = float(raw_logits.to(dtype=torch.float64).square().mean().sqrt().item())
    threshold_rms = float(threshold.to(dtype=torch.float64).square().mean().sqrt().item())
    eligible = _threshold_rms_is_compliant(threshold, raw_logit_rms=raw_rms)
    decisions = _decision_from_raw(raw_logits, threshold, temperature)
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "kind": kind,
        "targets": int(threshold.numel()),
        "eligible": bool(eligible),
        "rejection_reason": "" if eligible else "threshold_rms_not_strictly_below_bound",
        "threshold": [float(value) for value in threshold.tolist()],
        "temperature": [float(value) for value in temperature.tolist()],
        "threshold_rms": threshold_rms,
        "raw_logit_rms": raw_rms,
        "rms_fraction": RMS_FRACTION,
        "metrics": _decision_ranking_metrics(raw_logits, labels, decisions),
        "decision_stats": _decision_metric_stats(decisions, labels),
        "search": resolved_search,
        "shrinkage": copy.deepcopy(shrinkage) if shrinkage is not None else None,
        "provenance": copy.deepcopy(_FAMILY_PROVENANCE[kind]),
    }


def _best_global_threshold(raw_logits: Tensor, labels: Tensor) -> float:
    best: tuple[float, float, float] | None = None
    for threshold in THRESHOLD_GRID:
        score = float(_f1_per_label(raw_logits > float(threshold), labels).mean().item())
        candidate = (float(score), -abs(threshold), -threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return -best[2]


def _best_threshold_for_labels(raw_logits: Tensor, labels: Tensor, label_indices: Tensor, temperature: float = 1.0) -> float:
    best: tuple[float, float, float] | None = None
    for threshold in THRESHOLD_GRID:
        values = raw_logits[:, label_indices] > float(temperature) * float(threshold)
        score = float(_f1_per_label(values, labels[:, label_indices]).mean().item())
        candidate = (score, -abs(threshold), -threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return -best[2]


def _compute_shrinkage_thresholds_float32(
    *,
    label_thresholds: Tensor,
    support_counts: Tensor,
    group_ids: Sequence[str],
    group_thresholds: Mapping[str, float],
) -> Tensor:
    """Canonical P15 float32 shrinkage computation shared by fit and validation."""

    if label_thresholds.ndim != 1 or support_counts.ndim != 1 or label_thresholds.shape != support_counts.shape:
        raise ValueError("shrinkage inputs must be matching one-dimensional tensors")
    if len(group_ids) != label_thresholds.numel():
        raise ValueError("shrinkage group IDs must match label thresholds")
    if not torch.is_floating_point(label_thresholds) or support_counts.dtype == torch.bool:
        raise TypeError("shrinkage threshold and support dtypes are invalid")
    device = label_thresholds.device
    labels32 = label_thresholds.detach().to(device=device, dtype=torch.float32)
    counts32 = support_counts.detach().to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(labels32).all()) or not bool(torch.isfinite(counts32).all()) or bool((counts32 < 0.0).any()):
        raise ValueError("shrinkage inputs must be finite with nonnegative support")
    try:
        group32 = torch.tensor([group_thresholds[group_id] for group_id in group_ids], device=device, dtype=torch.float32)
    except KeyError as error:
        raise ValueError("shrinkage group threshold is missing") from error
    if not bool(torch.isfinite(group32).all()):
        raise ValueError("shrinkage group thresholds must be finite")
    strength32 = torch.tensor(SHRINKAGE_STRENGTH, device=device, dtype=torch.float32)
    return (counts32 * labels32 + strength32 * group32) / (counts32 + strength32)


def _build_candidates(raw_logits: Tensor, labels: Tensor, groups: list[str], ordered_groups: list[str]) -> list[dict[str, Any]]:
    targets = raw_logits.shape[1]
    ones = torch.ones(targets, dtype=torch.float32)
    global_threshold = _best_global_threshold(raw_logits, labels)
    global_values = torch.full((targets,), global_threshold, dtype=torch.float32)
    global_candidate = _candidate_record(
        kind="global_threshold",
        raw_logits=raw_logits,
        labels=labels,
        threshold=global_values,
        temperature=ones,
        search={"executed": True, "threshold_grid": list(THRESHOLD_GRID), "tie_break": "mf1,abs_threshold,lower_threshold"},
    )

    group_thresholds: dict[str, float] = {}
    group_values = torch.empty(targets, dtype=torch.float32)
    for group in ordered_groups:
        indices = torch.tensor([index for index, value in enumerate(groups) if value == group], dtype=torch.long)
        threshold = _best_threshold_for_labels(raw_logits, labels, indices)
        group_thresholds[group] = threshold
        group_values[indices] = threshold
    group_candidate = _candidate_record(
        kind="group_threshold",
        raw_logits=raw_logits,
        labels=labels,
        threshold=group_values,
        temperature=ones,
        search={"executed": True, "threshold_grid": list(THRESHOLD_GRID), "group_ids": list(groups), "tie_break": "group_mean_f1,abs_threshold,lower_threshold"},
    )
    group_candidate["group_thresholds"] = {key: float(value) for key, value in group_thresholds.items()}

    label_thresholds = torch.empty(targets, dtype=torch.float32)
    support_counts = labels.sum(dim=0).to(dtype=torch.int64)
    for index in range(targets):
        label_thresholds[index] = _best_threshold_for_labels(raw_logits, labels, torch.tensor([index], dtype=torch.long))
    shrinkage_values = _compute_shrinkage_thresholds_float32(
        label_thresholds=label_thresholds,
        support_counts=support_counts,
        group_ids=groups,
        group_thresholds=group_thresholds,
    )
    shrink_candidate = _candidate_record(
        kind="shrinkage_per_label_threshold",
        raw_logits=raw_logits,
        labels=labels,
        threshold=shrinkage_values,
        temperature=ones,
        search={
            "executed": True,
            "threshold_grid": list(THRESHOLD_GRID),
            "group_ids": list(groups),
            "tie_break": "per_label_f1,abs_threshold,lower_threshold",
        },
        shrinkage={
            "support_counts": [int(value) for value in support_counts.tolist()],
            "strength": SHRINKAGE_STRENGTH,
            "formula_version": SHRINKAGE_FORMULA_VERSION,
            "group_thresholds": {key: float(value) for key, value in group_thresholds.items()},
            "label_thresholds_before_shrinkage": [float(value) for value in label_thresholds.tolist()],
        },
    )

    temperatures = torch.empty(targets, dtype=torch.float32)
    temperature_thresholds = torch.empty(targets, dtype=torch.float32)
    for index in range(targets):
        best_temperature: tuple[float, float, float, float, float] | None = None
        for temperature in TEMPERATURE_GRID:
            threshold = _best_threshold_for_labels(
                raw_logits, labels, torch.tensor([index], dtype=torch.long), temperature=temperature
            )
            decisions = raw_logits[:, index] > float(temperature) * float(threshold)
            score = float(_f1_per_label(decisions.unsqueeze(1), labels[:, index : index + 1]).item())
            candidate = (score, -abs(temperature - 1.0), -abs(threshold), -threshold, temperature)
            if best_temperature is None or candidate > best_temperature:
                best_temperature = candidate
        assert best_temperature is not None
        temperatures[index] = best_temperature[4]
        temperature_thresholds[index] = -best_temperature[3]
    temperature_candidate = _candidate_record(
        kind="positive_temperature_threshold",
        raw_logits=raw_logits,
        labels=labels,
        threshold=temperature_thresholds,
        temperature=temperatures,
        search={
            "executed": True,
            "threshold_grid": list(THRESHOLD_GRID),
            "temperature_grid": list(TEMPERATURE_GRID),
            "tie_break": "per_label_f1,temperature_closest_to_one,abs_threshold,lower_threshold",
        },
    )
    return [global_candidate, group_candidate, shrink_candidate, temperature_candidate]


def _identity_candidate(
    *,
    targets: int,
    reason: str,
    raw_fixed_mf1: float,
    raw_logit_rms: float,
    raw_fixed_metrics: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "kind": "identity",
            "threshold": [0.0] * targets,
            "temperature": [1.0] * targets,
            "threshold_rms": 0.0,
            "raw_logit_rms": float(raw_logit_rms),
            "metrics": copy.deepcopy(dict(raw_fixed_metrics)) if raw_fixed_metrics is not None else {"mf1": float(raw_fixed_mf1)},
            "eligible": True,
        },
        {"used": True, "reason": reason, "path": "identity"},
    )


def _select_safe_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    raw_fixed_mf1: float,
    targets: int,
    raw_fixed_metrics: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select primary candidates first, then an independent guarded global fallback."""

    if len(candidates) != len(CANDIDATE_KINDS):
        raise ValueError("exactly four calibration candidates are required")
    records = {str(candidate.get("kind")): copy.deepcopy(dict(candidate)) for candidate in candidates}
    if set(records) != set(CANDIDATE_KINDS):
        raise ValueError("candidate kinds must be exactly the P15 families")
    raw_logit_rms = float(records["global_threshold"].get("raw_logit_rms", 0.0))
    primary_order = {kind: index for index, kind in enumerate(PRIMARY_CANDIDATE_KINDS)}
    primary = [records[kind] for kind in PRIMARY_CANDIDATE_KINDS if bool(records[kind].get("eligible", False))]
    primary.sort(
        key=lambda candidate: (-float(candidate["metrics"]["mf1"]), primary_order[str(candidate["kind"])])
    )
    if primary and float(primary[0]["metrics"]["mf1"]) >= raw_fixed_mf1 - MAX_RAW_MF1_DROP:
        return primary[0], {"used": False, "reason": "primary_candidate_passed_guard", "path": "primary"}

    global_candidate = records["global_threshold"]
    if bool(global_candidate.get("eligible", False)) and float(global_candidate["metrics"]["mf1"]) >= raw_fixed_mf1 - MAX_RAW_MF1_DROP:
        return global_candidate, {"used": True, "reason": "primary_candidate_failed_guard", "path": "global_fallback"}
    return _identity_candidate(
        targets=targets,
        reason="global_fallback_illegal_or_failed_guard",
        raw_fixed_mf1=raw_fixed_mf1,
        raw_logit_rms=raw_logit_rms,
        raw_fixed_metrics=raw_fixed_metrics,
    )


def _canonical_payload(result: Mapping[str, Any]) -> str:
    if not isinstance(result, Mapping):
        raise TypeError("calibration result must be a mapping")
    payload = copy.deepcopy(dict(result))
    payload.pop(DIGEST_FIELD, None)
    try:
        return json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except ValueError as error:
        raise ValueError("calibration payload contains non-finite numeric values") from error


def _payload_sha256(result: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(result).encode("utf-8")).hexdigest()


def _with_payload_digest(result: Mapping[str, Any]) -> dict[str, Any]:
    frozen = json.loads(_canonical_payload(result))
    frozen[DIGEST_FIELD] = _payload_sha256(frozen)
    return frozen


def _require_numeric_vector(
    name: str,
    values: Any,
    *,
    targets: int,
    positive: bool = False,
    allowed_values: Sequence[float] | None = None,
) -> list[float]:
    if not isinstance(values, list) or len(values) != targets:
        raise ValueError(f"{name} must be a list with one value per target")
    resolved: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain finite numeric values")
        value_float = float(value)
        if not math.isfinite(value_float) or (positive and value_float <= 0.0):
            raise ValueError(f"{name} must contain finite{' positive' if positive else ''} values")
        if allowed_values is not None and value_float not in {float(allowed) for allowed in allowed_values}:
            raise ValueError(f"{name} must use a value from the declared allowed grid")
        resolved.append(value_float)
    return resolved


def _require_finite_scalar(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    resolved = float(value)
    if not math.isfinite(resolved) or (nonnegative and resolved < 0.0):
        raise ValueError(f"{name} must be finite{' and nonnegative' if nonnegative else ''}")
    return resolved


def _require_unit_interval_scalar(name: str, value: Any) -> float:
    resolved = _require_finite_scalar(name, value)
    if resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return resolved


def _validate_metrics(metrics: Any, *, targets: int, context: str) -> None:
    if not isinstance(metrics, Mapping) or metrics.get("ranking_source") != "raw_logits":
        raise ValueError(f"{context} metrics must use raw_logits ranking")
    expected_keys = {"mf1", "per_label_f1", "map", "per_label_ap", "ranking_source"}
    if set(metrics) != expected_keys:
        raise ValueError(f"{context} metrics schema is invalid")
    mf1 = _require_unit_interval_scalar(f"{context} metrics mf1", metrics.get("mf1"))
    mean_ap = _require_unit_interval_scalar(f"{context} metrics map", metrics.get("map"))
    per_label_f1 = _require_numeric_vector(f"{context} metrics per_label_f1", metrics.get("per_label_f1"), targets=targets)
    per_label_ap = _require_numeric_vector(f"{context} metrics per_label_ap", metrics.get("per_label_ap"), targets=targets)
    if any(value < 0.0 or value > 1.0 for value in (*per_label_f1, *per_label_ap)):
        raise ValueError(f"{context} metrics must be in [0,1]")
    if not math.isclose(mf1, sum(per_label_f1) / targets, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError(f"{context} metrics mf1 is inconsistent with per-label F1")
    if not math.isclose(mean_ap, sum(per_label_ap) / targets, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError(f"{context} metrics mAP is inconsistent with per-label AP")


def _is_canonical_group_key(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("int:"):
        suffix = value[4:]
        return bool(re.fullmatch(r"-?(?:0|[1-9][0-9]*)", suffix))
    if value.startswith("str:"):
        return bool(_SAFE_GROUP_ID.fullmatch(value[4:]))
    return False


def _validate_source_descriptor(source: Any) -> None:
    if not isinstance(source, Mapping):
        raise ValueError("source descriptor is invalid")
    split_hash = source.get("split_hash")
    if not isinstance(split_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", split_hash):
        raise ValueError("source split hash is invalid")
    count = source.get("stable_id_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("source stable ID count is invalid")
    provided_split_hash = source.get("provided_split_hash")
    if provided_split_hash is not None and (not isinstance(provided_split_hash, str) or not provided_split_hash):
        raise ValueError("source provided split hash is invalid")
    if count:
        if source.get("canonicalization") != "P12_canonicalize_sample_id" or source.get("canonical_id_order") != "input_row_order":
            raise ValueError("source stable IDs must use P12 canonical ordered identities")
    elif source.get("canonicalization") != "provided_split_hash_only" or source.get("canonical_id_order") != "not_available":
        raise ValueError("source descriptor canonicalization is invalid")


def _validate_decision_metric_stats(stats: Any, *, targets: int) -> tuple[list[float], list[int]]:
    if not isinstance(stats, Mapping):
        raise ValueError("candidate decision statistics are invalid")
    sample_count = stats.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("candidate decision statistics sample count is invalid")
    fields = ("true_positive", "false_positive", "false_negative", "true_negative", "support")
    resolved: dict[str, list[int]] = {}
    for field in fields:
        values = stats.get(field)
        if not isinstance(values, list) or len(values) != targets or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError(f"candidate decision statistics {field} are invalid")
        resolved[field] = values
    per_label_f1: list[float] = []
    for index in range(targets):
        tp = resolved["true_positive"][index]
        fp = resolved["false_positive"][index]
        fn = resolved["false_negative"][index]
        tn = resolved["true_negative"][index]
        support = resolved["support"][index]
        if support != tp + fn or tp + fp + fn + tn != sample_count:
            raise ValueError("candidate decision statistics do not form a valid confusion matrix")
        denominator = 2 * tp + fp + fn
        per_label_f1.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return per_label_f1, list(resolved["support"])


def _close_vectors(left: Sequence[float], right: Sequence[float], *, tolerance: float = 1.0e-7) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance) for a, b in zip(left, right)
    )


def _validate_candidate_provenance(candidate: Mapping[str, Any], *, kind: str, targets: int, threshold_values: list[float], temperature_values: list[float]) -> None:
    if candidate.get("provenance") != _FAMILY_PROVENANCE[kind]:
        raise ValueError("candidate family provenance is invalid")
    if kind == "global_threshold":
        if len(set(threshold_values)) != 1 or temperature_values != [1.0] * targets:
            raise ValueError("global candidate must use one shared identity-temperature threshold")
        if candidate.get("group_thresholds") is not None:
            raise ValueError("global candidate cannot include group thresholds")
        return
    search = candidate["search"]
    group_ids = search.get("group_ids")
    if kind == "group_threshold":
        if temperature_values != [1.0] * targets:
            raise ValueError("group candidate must use identity temperature")
        group_thresholds = candidate.get("group_thresholds")
        if not isinstance(group_thresholds, Mapping) or set(group_thresholds) != set(group_ids):
            raise ValueError("group candidate threshold provenance is invalid")
        for index, group_key in enumerate(group_ids):
            value = _require_finite_scalar("group candidate threshold", group_thresholds[group_key])
            if not math.isclose(threshold_values[index], value, rel_tol=0.0, abs_tol=1.0e-7):
                raise ValueError("group candidate threshold is inconsistent within its typed group")
        return
    if kind == "shrinkage_per_label_threshold":
        if temperature_values != [1.0] * targets:
            raise ValueError("shrinkage candidate must use identity temperature")
        shrinkage = candidate["shrinkage"]
        counts = shrinkage["support_counts"]
        strength = _require_finite_scalar("shrinkage strength", shrinkage["strength"], nonnegative=True)
        if strength != SHRINKAGE_STRENGTH or shrinkage.get("formula_version") != SHRINKAGE_FORMULA_VERSION:
            raise ValueError("shrinkage formula provenance is invalid")
        raw_thresholds = _require_numeric_vector(
            "shrinkage label thresholds", shrinkage["label_thresholds_before_shrinkage"], targets=targets
        )
        group_thresholds = shrinkage["group_thresholds"]
        for group_key in group_ids:
            _require_finite_scalar("shrinkage group threshold", group_thresholds[group_key])
        expected = _compute_shrinkage_thresholds_float32(
            label_thresholds=torch.tensor(raw_thresholds, dtype=torch.float32),
            support_counts=torch.tensor(counts, dtype=torch.int64),
            group_ids=group_ids,
            group_thresholds={str(key): float(value) for key, value in group_thresholds.items()},
        )
        observed = torch.tensor(threshold_values, dtype=torch.float32)
        if not torch.equal(observed, expected):
            raise ValueError("shrinkage candidate threshold is inconsistent with its recorded float32 formula")
        return
    if kind == "positive_temperature_threshold":
        if candidate.get("group_thresholds") is not None or candidate.get("shrinkage") is not None:
            raise ValueError("temperature candidate cannot include group or shrinkage provenance")
        return
    raise ValueError("unknown candidate family provenance")


def _validate_candidate_record(
    candidate: Mapping[str, Any],
    *,
    kind: str,
    targets: int,
    raw_logit_rms: float,
    raw_metrics: Mapping[str, Any],
) -> None:
    if not isinstance(candidate, Mapping) or candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION or candidate.get("kind") != kind:
        raise ValueError("candidate schema or kind is invalid")
    if candidate.get("targets") != targets:
        raise ValueError("candidate target count is invalid")
    threshold_values = _require_numeric_vector("candidate threshold", candidate.get("threshold"), targets=targets)
    temperature_values = _require_numeric_vector(
        "candidate temperature",
        candidate.get("temperature"),
        targets=targets,
        positive=True,
        allowed_values=TEMPERATURE_GRID,
    )
    threshold = torch.tensor(threshold_values, dtype=torch.float64)
    actual_rms = float(threshold.square().mean().sqrt().item())
    stored_rms = _require_finite_scalar("candidate threshold_rms", candidate.get("threshold_rms"), nonnegative=True)
    stored_raw_rms = _require_finite_scalar("candidate raw_logit_rms", candidate.get("raw_logit_rms"), nonnegative=True)
    if not math.isclose(stored_raw_rms, raw_logit_rms, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("candidate raw_logit_rms does not match result")
    if not math.isclose(stored_rms, actual_rms, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("candidate threshold_rms does not match threshold")
    expected_eligible = _threshold_rms_is_compliant(threshold, raw_logit_rms=stored_raw_rms)
    if candidate.get("eligible") is not expected_eligible:
        raise ValueError("candidate eligibility does not match strict RMS guard")
    search = candidate.get("search")
    if not isinstance(search, Mapping) or search.get("executed") is not True:
        raise ValueError("candidate search metadata is invalid")
    if search.get("threshold_grid") != list(THRESHOLD_GRID) or search.get("temperature_grid") != list(TEMPERATURE_GRID):
        raise ValueError("candidate search grid is invalid")
    group_ids = search.get("group_ids")
    if kind in {"group_threshold", "shrinkage_per_label_threshold"}:
        if not isinstance(group_ids, list) or len(group_ids) != targets or not all(_is_canonical_group_key(value) for value in group_ids):
            raise ValueError("candidate group IDs must use typed canonical namespaces")
    _validate_metrics(candidate.get("metrics"), targets=targets, context="candidate")
    metric_stats_f1, metric_stats_support = _validate_decision_metric_stats(candidate.get("decision_stats"), targets=targets)
    metrics = candidate["metrics"]
    recorded_f1 = _require_numeric_vector("candidate metrics per_label_f1", metrics["per_label_f1"], targets=targets)
    if not _close_vectors(metric_stats_f1, recorded_f1):
        raise ValueError("candidate metrics F1 is inconsistent with recorded decision statistics")
    if not math.isclose(float(metrics["mf1"]), sum(metric_stats_f1) / targets, rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError("candidate macro F1 is inconsistent with recorded decision statistics")
    if not _close_vectors(metrics["per_label_ap"], raw_metrics["per_label_ap"]) or not math.isclose(
        float(metrics["map"]), float(raw_metrics["map"]), rel_tol=0.0, abs_tol=1.0e-7
    ):
        raise ValueError("candidate ranking metrics must match frozen raw-logit ranking metrics")
    shrinkage = candidate.get("shrinkage")
    if kind == "shrinkage_per_label_threshold":
        if not isinstance(shrinkage, Mapping):
            raise ValueError("shrinkage candidate must include shrinkage metadata")
        support_counts = shrinkage.get("support_counts")
        if not isinstance(support_counts, list) or len(support_counts) != targets or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in support_counts
        ):
            raise ValueError("shrinkage support counts are invalid")
        if support_counts != metric_stats_support:
            raise ValueError("shrinkage support counts must exactly match candidate decision-statistics support")
        if _require_finite_scalar("shrinkage strength", shrinkage.get("strength"), nonnegative=True) != SHRINKAGE_STRENGTH:
            raise ValueError("shrinkage strength is invalid")
        _require_numeric_vector("shrinkage label thresholds", shrinkage.get("label_thresholds_before_shrinkage"), targets=targets)
        group_thresholds = shrinkage.get("group_thresholds")
        if not isinstance(group_thresholds, Mapping) or set(group_thresholds) != set(group_ids):
            raise ValueError("shrinkage group thresholds do not match typed group IDs")
        for group_key, value in group_thresholds.items():
            if not _is_canonical_group_key(group_key):
                raise ValueError("shrinkage group threshold key is invalid")
            _require_finite_scalar("shrinkage group threshold", value)
    elif shrinkage is not None:
        raise ValueError("only the shrinkage candidate may include shrinkage metadata")
    _validate_candidate_provenance(
        candidate,
        kind=kind,
        targets=targets,
        threshold_values=threshold_values,
        temperature_values=temperature_values,
    )


def _validate_result_integrity(result: Mapping[str, Any], *, targets: int) -> None:
    if not isinstance(result, Mapping) or result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result must be a P15 calibration result")
    if result.get("fit_split") != "train_calib" or result.get("targets") != targets:
        raise ValueError("result target count or fitted split is invalid")
    if result.get("allowed_temperature_grid") != list(TEMPERATURE_GRID):
        raise ValueError("result allowed temperature grid is invalid")
    if result.get("integrity") != {"model": INTEGRITY_MODEL, "sha256_limitation": SHA256_LIMITATION}:
        raise ValueError("result integrity claim is invalid")
    digest = result.get(DIGEST_FIELD)
    if not isinstance(digest, str) or len(digest) != 64 or digest != _payload_sha256(result):
        raise ValueError("calibration result payload digest mismatch")
    raw_logit_rms = _require_finite_scalar("raw_logit_rms", result.get("raw_logit_rms"), nonnegative=True)
    _validate_source_descriptor(result.get("source"))
    train_metrics = result.get("train_calib_metrics")
    if not isinstance(train_metrics, Mapping) or set(train_metrics) != {"raw_fixed", "deploy"}:
        raise ValueError("result must retain raw_fixed and deploy train-calib metrics")
    raw_fixed = train_metrics["raw_fixed"]
    deploy_metrics = train_metrics["deploy"]
    _validate_metrics(raw_fixed, targets=targets, context="raw_fixed")
    _validate_metrics(deploy_metrics, targets=targets, context="deploy")
    raw_fixed_mf1 = _require_unit_interval_scalar("raw_fixed mf1", raw_fixed.get("mf1"))
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or [candidate.get("kind") if isinstance(candidate, Mapping) else None for candidate in candidates] != list(CANDIDATE_KINDS):
        raise ValueError("candidate table must contain exactly four ordered unique families")
    by_kind: dict[str, Mapping[str, Any]] = {}
    for kind, candidate in zip(CANDIDATE_KINDS, candidates):
        _validate_candidate_record(
            candidate,
            kind=kind,
            targets=targets,
            raw_logit_rms=raw_logit_rms,
            raw_metrics=raw_fixed,
        )
        by_kind[kind] = candidate
    chosen = result.get("chosen")
    fallback = result.get("fallback")
    if not isinstance(chosen, Mapping) or not isinstance(fallback, Mapping):
        raise ValueError("result must contain chosen calibration and fallback metadata")
    chosen_kind = chosen.get("kind")
    if chosen_kind == "identity":
        if not bool(fallback.get("used")):
            raise ValueError("identity calibration requires an active fallback")
        if _require_numeric_vector("identity threshold", chosen.get("threshold"), targets=targets) != [0.0] * targets:
            raise ValueError("identity threshold must be exactly zero")
        if _require_numeric_vector(
            "identity temperature", chosen.get("temperature"), targets=targets, positive=True, allowed_values=TEMPERATURE_GRID
        ) != [1.0] * targets:
            raise ValueError("identity temperature must be exactly one")
    else:
        if chosen_kind not in by_kind:
            raise ValueError("chosen kind is not a P15 candidate")
        if not bool(by_kind[chosen_kind].get("eligible")):
            raise ValueError("chosen candidate must be eligible")
        if json.dumps(dict(chosen), allow_nan=False, sort_keys=True, separators=(",", ":")) != json.dumps(dict(by_kind[chosen_kind]), allow_nan=False, sort_keys=True, separators=(",", ":")):
            raise ValueError("chosen calibration must exactly match its frozen candidate")
    if not isinstance(chosen.get("metrics"), Mapping):
        raise ValueError("chosen calibration must retain complete metrics")
    _validate_metrics(chosen["metrics"], targets=targets, context="chosen")
    canonical_metrics = lambda value: json.dumps(dict(value), allow_nan=False, sort_keys=True, separators=(",", ":"))
    if canonical_metrics(chosen["metrics"]) != canonical_metrics(deploy_metrics):
        raise ValueError("chosen metrics must exactly match top-level deploy metrics")
    expected_chosen, expected_fallback = _select_safe_candidate(
        candidates=candidates,
        raw_fixed_mf1=raw_fixed_mf1,
        targets=targets,
        raw_fixed_metrics=raw_fixed,
    )
    canonical = lambda value: json.dumps(dict(value), allow_nan=False, sort_keys=True, separators=(",", ":"))
    if canonical(chosen) != canonical(expected_chosen) or canonical(fallback) != canonical(expected_fallback):
        raise ValueError("chosen calibration or fallback violates hierarchical selection")


@torch.no_grad()
def fit_posthoc_calibration(
    *,
    raw_logits: Tensor,
    labels: Tensor,
    split: str,
    group_ids: Sequence[int | str],
    stable_ids: Sequence[str | int] | None = None,
    split_hash: str | None = None,
) -> dict[str, Any]:
    """Fit exactly four train-calib-only candidate families on frozen logits."""

    if split != "train_calib":
        raise ValueError("fit split must be exactly train_calib")
    _require_cpu_float32("raw_logits", raw_logits)
    _require_binary_labels(labels, shape=raw_logits.shape)
    groups, ordered_groups = _normalize_groups(group_ids, targets=raw_logits.shape[1])
    source = _source_descriptor(stable_ids=stable_ids, split_hash=split_hash, batch=raw_logits.shape[0])
    frozen_logits = raw_logits.detach().clone()
    frozen_labels = labels.detach().clone()
    raw_metrics = multi_label_metrics(frozen_logits, frozen_labels)
    raw_metrics["ranking_source"] = "raw_logits"
    raw_logit_rms = float(frozen_logits.to(dtype=torch.float64).square().mean().sqrt().item())
    candidates = _build_candidates(frozen_logits, frozen_labels, groups, ordered_groups)
    chosen, fallback = _select_safe_candidate(
        candidates=candidates,
        raw_fixed_mf1=float(raw_metrics["mf1"]),
        targets=frozen_logits.shape[1],
        raw_fixed_metrics=raw_metrics,
    )
    if chosen["kind"] == "identity":
        deployed_metrics = copy.deepcopy(raw_metrics)
    else:
        threshold = torch.tensor(chosen["threshold"], dtype=torch.float32)
        temperature = torch.tensor(chosen["temperature"], dtype=torch.float32)
        deployed_metrics = _candidate_metrics(frozen_logits, frozen_labels, threshold, temperature)
    result = _with_payload_digest(
        {
            "schema_version": SCHEMA_VERSION,
            "fit_split": "train_calib",
            "targets": int(frozen_logits.shape[1]),
            "allowed_temperature_grid": list(TEMPERATURE_GRID),
            "integrity": {"model": INTEGRITY_MODEL, "sha256_limitation": SHA256_LIMITATION},
            "raw_logit_rms": raw_logit_rms,
            "source": source,
            "candidates": candidates,
            "chosen": chosen,
            "train_calib_metrics": {"raw_fixed": raw_metrics, "deploy": deployed_metrics},
            "fallback": fallback,
        }
    )
    _validate_result_integrity(result, targets=frozen_logits.shape[1])
    return result


def _deployment_parameters(result: Mapping[str, Any], *, targets: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    _validate_result_integrity(result, targets=targets)
    chosen = result.get("chosen")
    assert isinstance(chosen, Mapping)
    threshold_values = chosen.get("threshold")
    temperature_values = chosen.get("temperature")
    if not isinstance(threshold_values, list) or not isinstance(temperature_values, list):
        raise ValueError("result deployment parameters must be lists")
    if len(threshold_values) != targets or len(temperature_values) != targets:
        raise ValueError("result deployment parameter length mismatch")
    threshold = torch.tensor(threshold_values, device=device, dtype=torch.float32).detach()
    temperature = torch.tensor(temperature_values, device=device, dtype=torch.float32).detach()
    if not bool(torch.isfinite(threshold).all()) or not bool(torch.isfinite(temperature).all()) or not bool((temperature > 0.0).all()):
        raise ValueError("result must contain finite positive temperatures and finite thresholds")
    return threshold.to(dtype=dtype), temperature.to(dtype=dtype)


def apply_posthoc_calibration(raw_logits: Tensor, result: Mapping[str, Any]) -> dict[str, Tensor | str]:
    """Return immutable ranking, strict decisions, and float64 diagnostic margins.

    Ranking metrics must always consume ``ranking_logits``.  The decision route
    is intentionally not encoded as a transformed logit, avoiding nearby
    float32 ordering changes and accidental deploy-AP substitution.
    """

    if not isinstance(raw_logits, Tensor) or raw_logits.ndim != 2 or raw_logits.shape[1] not in TARGET_COUNTS:
        raise ValueError("raw_logits must be [B,4] or [B,21]")
    if not torch.is_floating_point(raw_logits):
        raise TypeError("raw_logits must be floating point")
    if not bool(torch.isfinite(raw_logits.detach()).all()):
        raise ValueError("raw_logits must contain only finite values")
    threshold, temperature = _deployment_parameters(
        result, targets=raw_logits.shape[1], device=raw_logits.device, dtype=torch.float32
    )
    ranking_logits = raw_logits.detach().clone()
    raw64 = raw_logits.detach().to(dtype=torch.float64)
    threshold64 = threshold.to(dtype=torch.float64)
    temperature64 = temperature.to(dtype=torch.float64)
    diagnostic_margin = raw64 - temperature64.unsqueeze(0) * threshold64.unsqueeze(0)
    return {
        "ranking_logits": ranking_logits,
        "decision": raw64 > temperature64.unsqueeze(0) * threshold64.unsqueeze(0),
        "diagnostic_margin": diagnostic_margin,
        "ranking_source": "raw_logits",
    }


def serialize_calibration_result(result: Mapping[str, Any]) -> str:
    """Serialize only a validated, digest-bound frozen result."""

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    targets = result.get("targets")
    if isinstance(targets, bool) or not isinstance(targets, int) or targets not in TARGET_COUNTS:
        raise ValueError("result target count is invalid")
    _validate_result_integrity(result, targets=targets)
    return json.dumps(copy.deepcopy(dict(result)), allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def deserialize_calibration_result(payload: str) -> dict[str, Any]:
    if not isinstance(payload, str):
        raise TypeError("payload must be a JSON string")
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError("payload is not a P15 calibration result")
    targets = result.get("targets")
    if isinstance(targets, bool) or not isinstance(targets, int) or targets not in TARGET_COUNTS:
        raise ValueError("payload target count is invalid")
    _validate_result_integrity(result, targets=targets)
    return copy.deepcopy(result)


@torch.no_grad()
def diagnostic_test_oracle(*, raw_logits: Tensor, labels: Tensor) -> dict[str, Any]:
    """Test-only diagnostic; it cannot produce a deployable fit result."""

    _require_cpu_float32("raw_logits", raw_logits)
    _require_binary_labels(labels, shape=raw_logits.shape)
    return {"kind": "test_oracle_diagnostic", "metrics": multi_label_metrics(raw_logits, labels)}


__all__ = [
    "apply_posthoc_calibration",
    "deserialize_calibration_result",
    "diagnostic_test_oracle",
    "fit_posthoc_calibration",
    "multi_label_metrics",
    "serialize_calibration_result",
]
