"""Strict, transaction-safe persistence owner for RAEL run and epoch artifacts.

This module is intentionally the only place that converts public trainer/evaluator
outputs into durable RAEL artifacts.  It rejects partial schemas rather than
silently recording placeholder diagnostics that would later look trustworthy.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from threading import Lock
import time
from typing import Any, Iterator, Mapping, Sequence
import uuid

import torch
from torch import Tensor

from fate_oia.models.rael_oia_model import BRANCH_NAMES as MODEL_BRANCH_NAMES
import yaml


RUN_ROOT_FILES = (
    "run_manifest.json",
    "config_resolved.yaml",
    "source_fingerprint.json",
    "runtime_profile.json",
    "runtime_steps.jsonl",
    "selected_runtime_profile.json",
    "optimizer_owners.json",
    "loss_components.jsonl",
    "gradient_admission.jsonl",
    "mechanism_stats.jsonl",
    "metrics_summary.jsonl",
    "pu_audit.jsonl",
)
EPOCH_FILES = (
    "raw_metrics.json",
    "deploy_metrics.json",
    "branch_metrics.json",
    "per_action.json",
    "per_reason.json",
    "slot_stats.json",
    "layer_stats.json",
    "relation_stats.json",
    "contribution_stats.json",
    "named_latent_global.json",
    "gradient_admission.json",
    "pu_stats.json",
    "counterfactual.json",
    "calibration.json",
    "failure_cases.jsonl",
    "evidence_cases.jsonl",
    "logits_raw.pt",
    "logits_deploy.pt",
    "labels.pt",
)

_RUN_ROOT_SET = frozenset(RUN_ROOT_FILES)
_EPOCH_SET = frozenset(EPOCH_FILES)
_SCHEMA_VERSION = "rael-artifact-v1"
_PROVENANCE_FIELDS = (
    "schema_version",
    "producer",
    "source_fingerprint_sha256",
    "config_sha256",
)
_FINGERPRINT_FIELDS = (
    "fingerprint_schema",
    "phase",
    "complete",
    "groups",
    "file_status",
    "file_sha256",
    "missing_files",
    "group_hashes",
    "source_hash",
    "config_hash",
    "schema_hash",
    "required_files_hash",
)
_TRAINER_CONFIG_FIELDS = (
    "precision",
    "gradient_accumulation_steps",
    "total_optimizer_updates",
    "counterfactual_every_optimizer_updates",
    "owner_learning_rates",
    "weight_decay",
    "grad_clip_norm",
    "seed",
)
_TRAINER_CONFIG_REQUIRED = (
    "precision",
    "gradient_accumulation_steps",
    "total_optimizer_updates",
    "owner_learning_rates",
)
_P17_FINGERPRINT_SCHEMA = "rael-repository-fingerprint-v4"
_P17_FINGERPRINT_GROUPS = ("source", "test", "config", "schema", "skill", "script")
_OPTIMIZER_OWNER_ORDER = (
    "multilayer_field",
    "slot_ledger_core",
    "slot_attribute_heads",
    "action_category",
    "semantic_reason",
    "action_reason_bridge",
    "unary_contribution",
    "pairwise_relation",
    "reason_private",
    "pu_private",
)
_EXPECTED_OPTIMIZER_OWNERS = frozenset(_OPTIMIZER_OWNER_ORDER)
_PRIVATE_OPTIMIZER_OWNERS = frozenset({"reason_private", "pu_private"})
_MAIN_OWNER_LR = 2.0e-4
_PRIVATE_OWNER_LR = 3.0e-4
_WEIGHT_DECAY = 0.05
_BRANCH_NAMES = tuple(MODEL_BRANCH_NAMES)
_JSONL_STEP_FILES = frozenset(
    {
        "loss_components.jsonl",
        "gradient_admission.jsonl",
        "mechanism_stats.jsonl",
        "runtime_steps.jsonl",
    }
)
_THREAD_LOCKS: dict[str, Lock] = {}
_THREAD_LOCKS_GUARD = Lock()


def _p17_stable_hash(payload: Mapping[str, Any]) -> str:
    """Match P17's deterministic JSON hashing for a public fingerprint manifest."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _p17_manifest_entries_hash(
    *,
    namespace: str,
    phase: str,
    paths: Sequence[str],
    file_status: Mapping[str, str],
    file_sha256: Mapping[str, str | None],
) -> str:
    """Match P17's per-group and required-file fingerprint aggregation exactly."""

    return _p17_stable_hash(
        {
            "namespace": namespace,
            "schema": _P17_FINGERPRINT_SCHEMA,
            "phase": phase,
            "files": [
                {
                    "path": path,
                    "status": file_status[path],
                    "sha256": file_sha256[path],
                }
                for path in sorted(paths)
            ],
        }
    )


def _reject_symlinked_parent(path: Path, *, context: str) -> None:
    """Reject every existing path component before ``resolve`` can hide a redirect."""

    current = path.absolute()
    while True:
        if current.exists() and current.is_symlink():
            raise ValueError(f"{context} must not contain a symlinked parent component")
        if current.parent == current:
            return
        current = current.parent


def _json_safe(value: Any, *, context: str) -> Any:
    """Return a recursive JSON-safe copy while rejecting non-finite values."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} must contain only finite floats")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} mapping keys must be strings")
            output[key] = _json_safe(item, context=f"{context}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item, context=f"{context}[]") for item in value]
    raise TypeError(f"{context} contains a non-JSON-safe value {type(value).__name__}")


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{context} mapping keys must be strings")
    return value


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{context} must be a 64-character SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{context} must be a lowercase hexadecimal SHA256")
    return value


def _finite_scalar(value: Any, *, context: str) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1 or value.is_complex():
            raise ValueError(f"{context} must be a finite real scalar tensor")
        value = float(value.detach().float().cpu().item())
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite numeric scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{context} must be finite")
    return resolved


def _finite_float_mapping(value: Any, *, context: str) -> dict[str, float]:
    mapping = _require_mapping(value, context=context)
    return {
        name: _finite_scalar(item, context=f"{context}.{name}")
        for name, item in mapping.items()
    }


def _validate_provenance(
    value: Any,
    *,
    context: str,
    epoch: int | None = None,
    sample_count: int | None = None,
) -> dict[str, Any]:
    mapping = _require_mapping(value, context=context)
    missing = [field for field in _PROVENANCE_FIELDS if field not in mapping]
    if missing:
        raise ValueError(f"{context} is missing provenance fields: {missing}")
    if mapping["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(f"{context}.schema_version must equal {_SCHEMA_VERSION}")
    if not isinstance(mapping["producer"], str) or not mapping["producer"].strip():
        raise ValueError(f"{context}.producer must be a nonempty callsite string")
    _require_sha256(mapping["source_fingerprint_sha256"], context=f"{context}.source_fingerprint_sha256")
    _require_sha256(mapping["config_sha256"], context=f"{context}.config_sha256")
    resolved_epoch = mapping.get("epoch")
    if epoch is not None:
        if resolved_epoch != epoch:
            raise ValueError(f"{context}.epoch must equal the epoch transaction")
    elif resolved_epoch is not None and (isinstance(resolved_epoch, bool) or not isinstance(resolved_epoch, int)):
        raise ValueError(f"{context}.epoch must be an integer when present")
    resolved_count = mapping.get("sample_count")
    if sample_count is not None:
        if resolved_count != sample_count:
            raise ValueError(f"{context}.sample_count must match tensor sample_count")
    elif resolved_count is not None and (
        isinstance(resolved_count, bool) or not isinstance(resolved_count, int) or resolved_count <= 0
    ):
        raise ValueError(f"{context}.sample_count must be a positive integer when present")
    return {
        field: mapping[field]
        for field in (*_PROVENANCE_FIELDS, "epoch", "sample_count")
        if field in mapping
    }


def _require_nonempty_data(value: Mapping[str, Any], *, context: str) -> None:
    if "data" in value:
        data = _require_mapping(value["data"], context=f"{context}.data")
        if not data:
            raise ValueError(f"{context}.data is a placeholder empty mapping")


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], *, context: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")


def _finite_mapping(value: Any, *, context: str, nonempty: bool = True) -> Mapping[str, Any]:
    mapping = _require_mapping(value, context=context)
    if nonempty and not mapping:
        raise ValueError(f"{context} must be nonempty")
    for key, item in mapping.items():
        _finite_scalar(item, context=f"{context}.{key}")
    return mapping


def _matrix(value: Any, *, rows: int, columns: int, context: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != rows:
        raise ValueError(f"{context} must have {rows} rows")
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)) or len(row) != columns:
            raise ValueError(f"{context}[{row_index}] must have {columns} columns")
        for column_index, item in enumerate(row):
            _finite_scalar(item, context=f"{context}[{row_index}][{column_index}]")


def _metric_bundle(value: Any, *, context: str) -> None:
    mapping = _require_mapping(value, context=context)
    _require_fields(mapping, ("mF1", "oF1", "mAP", "AUC"), context=context)
    for field in ("mF1", "oF1", "mAP", "AUC"):
        _finite_scalar(mapping[field], context=f"{context}.{field}")


def _validate_metrics(value: Mapping[str, Any], *, context: str) -> None:
    metrics = _require_mapping(value.get("metrics"), context=f"{context}.metrics")
    for branch in ("action", "reason"):
        _metric_bundle(metrics.get(branch), context=f"{context}.metrics.{branch}")
    _finite_scalar(metrics.get("joint"), context=f"{context}.metrics.joint")


def _validate_per_label_rows(value: Any, *, expected: int, context: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{context} must be a sequence")
    if len(value) != expected:
        raise ValueError(f"{context} must contain exactly {expected} per-label rows")
    for index, row in enumerate(value):
        row_mapping = _require_mapping(row, context=f"{context}[{index}]")
        if row_mapping.get("id") != index or not isinstance(row_mapping.get("name"), str):
            raise ValueError(f"{context} must have ordered id/name fields")
        if not row_mapping["name"]:
            raise ValueError(f"{context}[{index}].name must be nonempty")
        for field in ("F1", "AP", "AUC", "support", "threshold"):
            _finite_scalar(row_mapping.get(field), context=f"{context}[{index}].{field}")


def _validate_epoch_json(name: str, value: Any, *, epoch: int) -> dict[str, Any]:
    mapping = _require_mapping(value, context=name)
    provenance = _validate_provenance(mapping, context=name, epoch=epoch)
    if "sample_count" not in provenance:
        raise ValueError(f"{name}.sample_count is required for epoch artifacts")
    if name in {"raw_metrics.json", "deploy_metrics.json"}:
        _validate_metrics(mapping, context=name)
    elif name == "branch_metrics.json":
        branches = mapping.get("branches")
        if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes, bytearray)):
            raise TypeError("branch_metrics.json.branches must be a sequence")
        if len(branches) != len(_BRANCH_NAMES):
            raise ValueError("branch_metrics.json must record exactly 14 branch metrics")
        names: list[str] = []
        for index, branch in enumerate(branches):
            branch_mapping = _require_mapping(branch, context=f"branch_metrics.json.branches[{index}]")
            if not isinstance(branch_mapping.get("name"), str) or not branch_mapping["name"]:
                raise ValueError("branch_metrics.json branch names must be nonempty")
            names.append(branch_mapping["name"])
            config = _require_mapping(branch_mapping.get("config"), context="branch_metrics.json.config")
            if config.get("diagnostic_mode") != branch_mapping["name"]:
                raise ValueError("branch_metrics.json.config must bind diagnostic_mode to branch name")
            branch_metrics = _require_mapping(branch_mapping.get("metrics"), context="branch_metrics.json.metrics")
            _require_fields(branch_metrics, ("action", "reason", "joint"), context="branch_metrics.json.metrics")
            _metric_bundle(branch_metrics["action"], context="branch_metrics.json.metrics.action")
            _metric_bundle(branch_metrics["reason"], context="branch_metrics.json.metrics.reason")
            _finite_scalar(branch_metrics["joint"], context="branch_metrics.json.metrics.joint")
            _require_fields(
                branch_mapping,
                ("per_action", "per_reason"),
                context=f"branch_metrics.json.branches[{index}]",
            )
            _validate_per_label_rows(
                branch_mapping.get("per_action"),
                expected=4,
                context=f"branch_metrics.json.branches[{index}].per_action",
            )
            _validate_per_label_rows(
                branch_mapping.get("per_reason"),
                expected=21,
                context=f"branch_metrics.json.branches[{index}].per_reason",
            )
        if tuple(names) != _BRANCH_NAMES:
            raise ValueError("branch_metrics.json branch names must match the formal 14-branch protocol")
    elif name in {"per_action.json", "per_reason.json"}:
        expected = 4 if name == "per_action.json" else 21
        _validate_per_label_rows(mapping.get("rows"), expected=expected, context=f"{name}.rows")
    elif name == "slot_stats.json":
        _require_fields(
            mapping,
            ("slot_count", "mass", "area", "entropy", "iou", "attributes", "reliability"),
            context=name,
        )
        if mapping["slot_count"] != 20:
            raise ValueError("slot_stats.json.slot_count must equal 20")
        _finite_mapping(mapping["mass"], context="slot_stats.json.mass")
        _finite_mapping(mapping["area"], context="slot_stats.json.area")
        _finite_scalar(mapping["entropy"], context="slot_stats.json.entropy")
        _finite_scalar(mapping["iou"], context="slot_stats.json.iou")
        _finite_mapping(mapping["attributes"], context="slot_stats.json.attributes")
        _finite_mapping(mapping["reliability"], context="slot_stats.json.reliability")
    elif name == "layer_stats.json":
        _require_fields(
            mapping,
            ("action_layer_weights", "reason_layer_weights", "slot_layer_weights", "entropy", "collapse"),
            context=name,
        )
        _matrix(mapping["action_layer_weights"], rows=4, columns=4, context="layer_stats.json.action_layer_weights")
        _matrix(mapping["reason_layer_weights"], rows=21, columns=4, context="layer_stats.json.reason_layer_weights")
        _matrix(mapping["slot_layer_weights"], rows=20, columns=4, context="layer_stats.json.slot_layer_weights")
        _finite_scalar(mapping["entropy"], context="layer_stats.json.entropy")
        if not isinstance(mapping["collapse"], bool):
            raise ValueError("layer_stats.json.collapse must be boolean")
    elif name == "relation_stats.json":
        _require_fields(
            mapping,
            ("unary", "pairwise", "null", "alpha", "active_pair_count", "total_pair_count"),
            context=name,
        )
        for field in ("unary", "pairwise", "null", "alpha"):
            _finite_mapping(mapping[field], context=f"relation_stats.json.{field}")
        for field in ("active_pair_count", "total_pair_count"):
            value_count = mapping[field]
            if isinstance(value_count, bool) or not isinstance(value_count, int) or value_count < 0:
                raise ValueError(f"relation_stats.json.{field} must be a nonnegative integer")
    elif name == "contribution_stats.json":
        _require_fields(
            mapping,
            ("global", "unary", "pairwise", "positive", "negative", "reconstruction_error"),
            context=name,
        )
        for field in ("global", "unary", "pairwise", "positive", "negative"):
            _finite_mapping(mapping[field], context=f"contribution_stats.json.{field}")
        _finite_scalar(mapping["reconstruction_error"], context="contribution_stats.json.reconstruction_error")
    elif name == "named_latent_global.json":
        _require_fields(
            mapping,
            ("named_ratio", "latent_ratio", "global_ratio", "per_target", "overall"),
            context=name,
        )
        for field in ("named_ratio", "latent_ratio", "global_ratio"):
            _finite_scalar(mapping[field], context=f"named_latent_global.json.{field}")
        per_target = mapping["per_target"]
        if not isinstance(per_target, Sequence) or isinstance(per_target, (str, bytes, bytearray)) or not per_target:
            raise ValueError("named_latent_global.json.per_target must be nonempty")
        for index, target in enumerate(per_target):
            target_mapping = _require_mapping(target, context=f"named_latent_global.json.per_target[{index}]")
            _require_fields(target_mapping, ("target", "named", "latent", "global"), context="named_latent_global.json.per_target")
            for field in ("named", "latent", "global"):
                _finite_scalar(target_mapping[field], context=f"named_latent_global.json.per_target.{field}")
        overall = _require_mapping(mapping["overall"], context="named_latent_global.json.overall")
        _require_fields(overall, ("named", "latent", "global"), context="named_latent_global.json.overall")
        for field in ("named", "latent", "global"):
            _finite_scalar(overall[field], context=f"named_latent_global.json.overall.{field}")
    elif name == "gradient_admission.json":
        _require_fields(mapping, ("cosine", "projection", "admission", "caps", "ema"), context=name)
        for field in ("cosine", "projection", "admission", "caps", "ema"):
            _finite_mapping(mapping[field], context=f"gradient_admission.json.{field}")
    elif name == "pu_stats.json":
        _require_fields(mapping, ("labels",), context=name)
        labels = mapping["labels"]
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes, bytearray)) or len(labels) != 21:
            raise ValueError("pu_stats.json.labels must contain exactly 21 rows")
        for index, label in enumerate(labels):
            label_mapping = _require_mapping(label, context=f"pu_stats.json.labels[{index}]")
            _require_fields(label_mapping, ("label_id", "gate", "score", "lambda", "soft_positive_count"), context="pu_stats.json.labels")
            if label_mapping["label_id"] != index or not isinstance(label_mapping["gate"], bool):
                raise ValueError("pu_stats.json.labels must have ordered ids and boolean gates")
            for field in ("score", "lambda", "soft_positive_count"):
                _finite_scalar(label_mapping[field], context=f"pu_stats.json.labels[{index}].{field}")
    elif name == "counterfactual.json":
        _require_fields(
            mapping,
            ("available", "reason", "sample_ids", "selected", "control", "wrong", "valid_action_target_count", "valid_reason_target_count"),
            context=name,
        )
        if not isinstance(mapping["available"], bool):
            raise ValueError("counterfactual.json.available must be boolean")
        if mapping["reason"] not in {"formal_128_case_audit", "no_eligible_control"}:
            raise ValueError("counterfactual.json.reason is not recognized")
        sample_ids = mapping["sample_ids"]
        if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes, bytearray)) or len(sample_ids) != 128:
            raise ValueError("counterfactual.json.sample_ids must contain exactly 128 fixed cases")
        if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
            raise ValueError("counterfactual.json.sample_ids must be nonempty strings")
        for field in ("selected", "control", "wrong"):
            effect = _require_mapping(mapping[field], context=f"counterfactual.json.{field}")
            value = effect.get("effect")
            if mapping["available"]:
                _finite_scalar(value, context=f"counterfactual.json.{field}.effect")
            elif value is not None:
                raise ValueError(f"counterfactual.json.{field}.effect must be null when unavailable")
        if mapping["available"] != (mapping["reason"] == "formal_128_case_audit"):
            raise ValueError("counterfactual.json availability and reason disagree")
        for field in ("valid_action_target_count", "valid_reason_target_count"):
            value_count = mapping[field]
            if isinstance(value_count, bool) or not isinstance(value_count, int) or value_count < 0:
                raise ValueError(f"counterfactual.json.{field} must be a nonnegative integer")
    elif name == "calibration.json":
        _require_fields(
            mapping,
            ("candidates", "chosen_thresholds", "temperature", "threshold_rms", "raw_map", "deploy_map", "fallback"),
            context=name,
        )
        candidates = mapping["candidates"]
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)) or len(candidates) != 4:
            raise ValueError("calibration.json.candidates must contain exactly four candidates")
        for candidate in candidates:
            candidate_mapping = _require_mapping(candidate, context="calibration.json.candidates")
            if not isinstance(candidate_mapping.get("name"), str) or not candidate_mapping["name"]:
                raise ValueError("calibration.json.candidates.name is required")
            _finite_scalar(candidate_mapping.get("joint"), context="calibration.json.candidates.joint")
        thresholds = _require_mapping(mapping["chosen_thresholds"], context="calibration.json.chosen_thresholds")
        if len(thresholds.get("action", ())) != 4 or len(thresholds.get("reason", ())) != 21:
            raise ValueError("calibration.json.chosen_thresholds must have action=4 and reason=21")
        for field in ("temperature", "threshold_rms", "raw_map", "deploy_map"):
            _finite_mapping(mapping[field], context=f"calibration.json.{field}")
        fallback = _require_mapping(mapping["fallback"], context="calibration.json.fallback")
        if not isinstance(fallback.get("used"), bool) or not isinstance(fallback.get("reason"), str):
            raise ValueError("calibration.json.fallback must contain used/reason")
    else:
        raise ValueError(f"unsupported epoch JSON schema: {name}")
    _json_safe(mapping, context=name)
    return provenance


def _tensor_safe(value: Any, *, context: str) -> Any:
    if isinstance(value, Tensor):
        if value.is_complex():
            raise TypeError(f"{context} tensor artifact must not use complex dtype")
        detached = value.detach()
        if (torch.is_floating_point(detached) or detached.is_complex()) and not bool(torch.isfinite(detached).all()):
            raise ValueError(f"{context} tensor artifact must contain only finite values")
        return detached.to(device="cpu").clone()
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} tensor artifact keys must be strings")
            output[key] = _tensor_safe(item, context=f"{context}.{key}")
        return output
    if isinstance(value, tuple):
        return tuple(_tensor_safe(item, context=f"{context}[]") for item in value)
    if isinstance(value, list):
        return [_tensor_safe(item, context=f"{context}[]") for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} tensor artifact must contain only finite values")
        return value
    raise TypeError(f"{context} contains unsupported tensor artifact value {type(value).__name__}")


def _validate_tensor(value: Any, *, context: str, shape: tuple[int, int], dtypes: frozenset[torch.dtype]) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{context} must be a tensor")
    if value.is_complex():
        raise TypeError(f"{context} must not use complex dtype")
    if value.dtype not in dtypes:
        allowed = "/".join(str(dtype).replace("torch.", "") for dtype in sorted(dtypes, key=str))
        raise TypeError(f"{context} must use {allowed}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{context} has shape {tuple(value.shape)}, expected {shape}")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{context} must contain only finite values")
    return value


def _validate_logits_tensor(name: str, value: Any, *, epoch: int) -> dict[str, Any]:
    mapping = _require_mapping(value, context=name)
    expected = {"_meta", "action", "reason"}
    if set(mapping) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")
    provenance = _validate_provenance(mapping["_meta"], context=f"{name}._meta", epoch=epoch)
    if "sample_count" not in provenance:
        raise ValueError(f"{name}._meta.sample_count is required")
    count = provenance["sample_count"]
    _validate_tensor(mapping["action"], context=f"{name}.action", shape=(count, 4), dtypes=frozenset({torch.float32}))
    _validate_tensor(mapping["reason"], context=f"{name}.reason", shape=(count, 21), dtypes=frozenset({torch.float32}))
    return provenance


def _validate_labels_tensor(value: Any, *, epoch: int) -> dict[str, Any]:
    mapping = _require_mapping(value, context="labels.pt")
    expected = {"_meta", "action", "reason", "file_names"}
    if set(mapping) != expected:
        raise ValueError(f"labels.pt must contain exactly {sorted(expected)}")
    provenance = _validate_provenance(mapping["_meta"], context="labels.pt._meta", epoch=epoch)
    if "sample_count" not in provenance:
        raise ValueError("labels.pt._meta.sample_count is required")
    count = provenance["sample_count"]
    for field, width in (("action", 4), ("reason", 21)):
        tensor = _validate_tensor(
            mapping[field],
            context=f"labels.pt.{field}",
            shape=(count, width),
            dtypes=frozenset({torch.uint8, torch.float32}),
        )
        if not bool(torch.logical_or(tensor == 0, tensor == 1).all()):
            raise ValueError(f"labels.pt.{field} must contain binary 0/1 labels")
    file_names = mapping["file_names"]
    if not isinstance(file_names, Sequence) or isinstance(file_names, (str, bytes, bytearray)):
        raise TypeError("labels.pt.file_names must be a sequence")
    if len(file_names) != count:
        raise ValueError("labels.pt.file_names length must equal sample_count")
    if any(not isinstance(item, str) or not item for item in file_names):
        raise ValueError("labels.pt.file_names must contain nonempty strings")
    return provenance


def _validate_epoch_jsonl(name: str, value: Any, *, epoch: int) -> dict[str, Any]:
    if isinstance(value, Mapping):
        rows = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = list(value)
    else:
        raise TypeError(f"{name} must contain a row mapping or row sequence")
    if not rows:
        raise ValueError(f"{name} must not be empty")
    sample_count: int | None = None
    canonical_provenance: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        mapping = _require_mapping(row, context=f"{name}[{index}]")
        provenance = _validate_provenance(mapping, context=f"{name}[{index}]", epoch=epoch)
        if "sample_count" not in provenance:
            raise ValueError(f"{name}[{index}].sample_count is required")
        if sample_count is None:
            sample_count = provenance["sample_count"]
        elif sample_count != provenance["sample_count"]:
            raise ValueError(f"{name} must use a single sample_count")
        if canonical_provenance is None:
            canonical_provenance = provenance
        elif any(
            provenance[field] != canonical_provenance[field]
            for field in _PROVENANCE_FIELDS
        ):
            raise ValueError(f"{name} rows must share one producer and provenance")
        if not isinstance(mapping.get("file_name"), str) or not mapping["file_name"]:
            raise ValueError(f"{name}[{index}].file_name is required")
        if not isinstance(mapping.get("case_id"), str) or not mapping["case_id"]:
            raise ValueError(f"{name}[{index}].case_id is required")
        data = _require_mapping(mapping.get("data"), context=f"{name}[{index}].data")
        if name == "failure_cases.jsonl":
            _require_fields(
                data,
                ("labels", "raw_predictions", "deploy_predictions", "branch_deltas"),
                context=f"{name}[{index}].data",
            )
            for field in ("labels", "raw_predictions", "deploy_predictions", "branch_deltas"):
                if not _require_mapping(data[field], context=f"{name}[{index}].data.{field}"):
                    raise ValueError(f"{name}[{index}].data.{field} must be nonempty")
        else:
            _require_fields(
                data,
                ("target", "selected_slots", "masks", "attributes", "contributions"),
                context=f"{name}[{index}].data",
            )
            target = _require_mapping(data["target"], context=f"{name}[{index}].data.target")
            _require_fields(target, ("type", "id"), context=f"{name}[{index}].data.target")
            if not isinstance(target["type"], str) or isinstance(target["id"], bool) or not isinstance(target["id"], int):
                raise ValueError(f"{name}[{index}].data.target must contain type/id")
            for field in ("selected_slots", "masks", "attributes", "contributions"):
                if not data[field]:
                    raise ValueError(f"{name}[{index}].data.{field} must be nonempty")
        _json_safe(mapping, context=f"{name}[{index}]")
    if canonical_provenance is None:
        raise AssertionError("nonempty epoch JSONL must establish provenance")
    return {**canonical_provenance, "row_count": len(rows), "sample_count": sample_count}


def _validate_p17_v4_fingerprint_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the public P17-v4 manifest by recomputing every derived hash."""

    if set(payload) != set(_FINGERPRINT_FIELDS):
        raise ValueError("fingerprint manifest fields are incomplete or unexpected")
    if payload["fingerprint_schema"] != _P17_FINGERPRINT_SCHEMA:
        raise ValueError("fingerprint manifest schema mismatch")
    phase = payload["phase"]
    if not isinstance(phase, str) or not phase or phase.strip() != phase:
        raise ValueError("fingerprint manifest phase must be nonempty normalized text")
    if not isinstance(payload["complete"], bool):
        raise ValueError("fingerprint manifest complete must be boolean")

    groups = _require_mapping(payload["groups"], context="fingerprint manifest groups")
    group_hashes = _require_mapping(
        payload["group_hashes"],
        context="fingerprint manifest group_hashes",
    )
    if set(groups) != set(_P17_FINGERPRINT_GROUPS) or set(group_hashes) != set(
        _P17_FINGERPRINT_GROUPS
    ):
        raise ValueError(
            "fingerprint manifest must cover P17 source/test/config/schema/skill/script groups"
        )

    normalized_groups: dict[str, list[str]] = {}
    all_paths: list[str] = []
    for group in _P17_FINGERPRINT_GROUPS:
        paths = groups[group]
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"fingerprint manifest group {group} must be a nonempty list")
        if any(not isinstance(path, str) or not path for path in paths):
            raise ValueError(f"fingerprint manifest group {group} paths must be nonempty text")
        normalized: list[str] = []
        for path in paths:
            candidate = Path(path)
            if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != path:
                raise ValueError(
                    f"fingerprint manifest group {group} contains a non-normalized relative path"
                )
            normalized.append(path)
        if normalized != sorted(normalized):
            raise ValueError(f"fingerprint manifest group {group} is not deterministically sorted")
        normalized_groups[group] = normalized
        all_paths.extend(normalized)
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("fingerprint manifest cannot assign one file to multiple groups")

    file_status = _require_mapping(payload["file_status"], context="fingerprint manifest file_status")
    file_sha256 = _require_mapping(payload["file_sha256"], context="fingerprint manifest file_sha256")
    if set(file_status) != set(all_paths) or set(file_sha256) != set(all_paths):
        raise ValueError("fingerprint manifest file_status/file_sha256 per-file coverage is invalid")
    for path in all_paths:
        status = file_status[path]
        digest = file_sha256[path]
        if status not in {"present", "missing"}:
            raise ValueError("fingerprint manifest file status is invalid")
        if status == "present":
            _require_sha256(digest, context=f"fingerprint manifest file_sha256.{path}")
        elif digest is not None:
            raise ValueError("fingerprint manifest missing files must have null SHA256")
    expected_missing = sorted(path for path in all_paths if file_status[path] == "missing")
    if payload["missing_files"] != expected_missing:
        raise ValueError("fingerprint manifest missing-file set is inconsistent")
    if payload["complete"] != (not expected_missing):
        raise ValueError("fingerprint manifest completeness is inconsistent")

    expected_group_hashes = {
        group: _p17_manifest_entries_hash(
            namespace=f"rael-{group}-v4",
            phase=phase,
            paths=normalized_groups[group],
            file_status=file_status,
            file_sha256=file_sha256,
        )
        for group in _P17_FINGERPRINT_GROUPS
    }
    if dict(group_hashes) != expected_group_hashes:
        raise ValueError("fingerprint manifest group hashes do not match P17-v4 recomputation")
    expected_source_hash = _p17_stable_hash(
        {
            "namespace": "rael-source-test-skill-script-v4",
            "phase": phase,
            "groups": {
                group: expected_group_hashes[group]
                for group in ("source", "test", "skill", "script")
            },
        }
    )
    expected_required_hash = _p17_manifest_entries_hash(
        namespace="rael-required-declared-and-import-closure-v4",
        phase=phase,
        paths=all_paths,
        file_status=file_status,
        file_sha256=file_sha256,
    )
    expected_aggregates = {
        "source_hash": expected_source_hash,
        "config_hash": expected_group_hashes["config"],
        "schema_hash": expected_group_hashes["schema"],
        "required_files_hash": expected_required_hash,
    }
    for field, expected in expected_aggregates.items():
        _require_sha256(payload[field], context=f"fingerprint manifest {field}")
        if payload[field] != expected:
            raise ValueError(f"fingerprint manifest {field} does not match P17-v4 recomputation")
    return _json_safe(dict(payload), context="fingerprint manifest")


def _validate_run_json(name: str, value: Any) -> dict[str, Any]:
    mapping = _require_mapping(value, context=name)
    provenance = _validate_provenance(mapping, context=name)
    if name == "run_manifest.json":
        required = {"git_head", "remote_head", "base_head", "command", "data_split", "dino", "formal_flags", "selected_runtime_profile", "seed", "test_selected", "publication_eligible"}
        missing = sorted(required.difference(mapping))
        if missing:
            raise ValueError(f"run_manifest.json is missing required fields: {missing}")
        for field in ("git_head", "remote_head", "base_head"):
            if not isinstance(mapping[field], str) or len(mapping[field]) != 40:
                raise ValueError(f"run_manifest.json.{field} must be a git SHA")
        formal = _require_mapping(mapping["formal_flags"], context="run_manifest.json.formal_flags")
        if formal.get("direct_image") is not True or formal.get("test_only") is not True:
            raise ValueError("run_manifest.json must preserve direct-image test-only protocol")
        if formal.get("feature_cache_enabled") is not False or formal.get("token_compression") != "none":
            raise ValueError("run_manifest.json must disable cache and token compression")
    elif name == "config_resolved.yaml":
        resolved_config = _require_mapping(mapping.get("resolved_config"), context="config_resolved.yaml.resolved_config")
        if not resolved_config:
            raise ValueError("config_resolved.yaml.resolved_config must not be empty")
        _require_sha256(mapping.get("resolved_config_sha256"), context="config_resolved.yaml.resolved_config_sha256")
        expected_config_hash = hashlib.sha256(
            json.dumps(_json_safe(resolved_config, context="config_resolved.yaml.resolved_config"), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if mapping["resolved_config_sha256"] != expected_config_hash:
            raise ValueError("config_resolved.yaml.resolved_config_sha256 must bind resolved_config")
        if mapping["config_sha256"] != expected_config_hash:
            raise ValueError("config_resolved.yaml.config_sha256 must bind resolved_config")
    elif name == "source_fingerprint.json":
        fingerprint = _validate_p17_v4_fingerprint_manifest(
            {field: mapping[field] for field in _FINGERPRINT_FIELDS if field in mapping}
        )
        if set(fingerprint) != set(_FINGERPRINT_FIELDS):
            raise AssertionError("P17 fingerprint validator must return the exact public schema")
        if mapping["required_files_hash"] != mapping["source_fingerprint_sha256"]:
            raise ValueError("source_fingerprint.json.required_files_hash must bind source_fingerprint_sha256")
    elif name == "runtime_profile.json":
        candidates = mapping.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)) or not candidates:
            raise ValueError("runtime_profile.json.candidates must be nonempty")
    elif name == "selected_runtime_profile.json":
        selected = _require_mapping(mapping.get("selected"), context="selected_runtime_profile.json.selected")
        for field in ("name", "batch_size", "gradient_accumulation_steps", "num_workers", "amortized_samples_per_sec"):
            if field not in selected:
                raise ValueError(f"selected_runtime_profile.json.selected.{field} is required")
        if not isinstance(mapping.get("reason"), str) or not mapping["reason"]:
            raise ValueError("selected_runtime_profile.json.reason is required")
    elif name == "optimizer_owners.json":
        owners = _require_mapping(mapping.get("owners"), context="optimizer_owners.json.owners")
        if set(owners) != _EXPECTED_OPTIMIZER_OWNERS:
            raise ValueError("optimizer_owners.json owner topology must contain exactly the ten formal owners")
        seen_parameters: set[str] = set()
        for owner in _OPTIMIZER_OWNER_ORDER:
            payload = owners[owner]
            owner_mapping = _require_mapping(payload, context=f"optimizer_owners.json.owners.{owner}")
            _require_fields(
                owner_mapping,
                ("parameter_names", "parameters", "lr", "weight_decay", "count"),
                context=f"optimizer_owners.json.owners.{owner}",
            )
            names = owner_mapping["parameter_names"]
            if not isinstance(names, Sequence) or isinstance(names, (str, bytes, bytearray)) or not names:
                raise ValueError(f"optimizer_owners.json.owners.{owner}.parameter_names must be nonempty")
            if any(not isinstance(parameter_name, str) or not parameter_name for parameter_name in names):
                raise ValueError(f"optimizer_owners.json.owners.{owner}.parameter_names must be strings")
            if len(names) != len(set(names)) or seen_parameters.intersection(names):
                raise ValueError("optimizer_owners.json parameter names must be globally unique")
            seen_parameters.update(names)
            if owner_mapping["count"] != len(names):
                raise ValueError(f"optimizer_owners.json.owners.{owner}.count must equal parameter_names length")
            expected_lr = _PRIVATE_OWNER_LR if owner in _PRIVATE_OPTIMIZER_OWNERS else _MAIN_OWNER_LR
            if _finite_scalar(owner_mapping["lr"], context=f"optimizer_owners.json.owners.{owner}.lr") != expected_lr:
                raise ValueError(f"optimizer_owners.json.owners.{owner}.lr violates the fixed owner LR contract")
            if _finite_scalar(owner_mapping["weight_decay"], context=f"optimizer_owners.json.owners.{owner}.weight_decay") != _WEIGHT_DECAY:
                raise ValueError(f"optimizer_owners.json.owners.{owner}.weight_decay must equal {_WEIGHT_DECAY}")
            parameters = owner_mapping["parameters"]
            if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes, bytearray)):
                raise TypeError(f"optimizer_owners.json.owners.{owner}.parameters must be a sequence")
            if len(parameters) != len(names):
                raise ValueError(f"optimizer_owners.json.owners.{owner}.parameters must match count")
            parameter_names: set[str] = set()
            for index, parameter in enumerate(parameters):
                parameter_mapping = _require_mapping(
                    parameter,
                    context=f"optimizer_owners.json.owners.{owner}.parameters[{index}]",
                )
                _require_fields(
                    parameter_mapping,
                    ("name", "lr", "weight_decay"),
                    context=f"optimizer_owners.json.owners.{owner}.parameters[{index}]",
                )
                parameter_name = parameter_mapping["name"]
                if not isinstance(parameter_name, str) or not parameter_name:
                    raise ValueError(f"optimizer_owners.json.owners.{owner}.parameters[{index}].name must be nonempty")
                parameter_names.add(parameter_name)
                if _finite_scalar(parameter_mapping["lr"], context=f"optimizer_owners.json.owners.{owner}.parameters[{index}].lr") != expected_lr:
                    raise ValueError(f"optimizer_owners.json.owners.{owner}.parameters[{index}].lr violates owner LR")
                decay = _finite_scalar(parameter_mapping["weight_decay"], context=f"optimizer_owners.json.owners.{owner}.parameters[{index}].weight_decay")
                if decay not in {0.0, _WEIGHT_DECAY}:
                    raise ValueError(f"optimizer_owners.json.owners.{owner}.parameters[{index}].weight_decay must be 0 or {_WEIGHT_DECAY}")
            if parameter_names != set(names):
                raise ValueError(f"optimizer_owners.json.owners.{owner}.parameters must cover parameter_names exactly")
    _json_safe(mapping, context=name)
    return provenance


def _validate_run_jsonl_row(name: str, row: Any) -> dict[str, Any]:
    mapping = _require_mapping(row, context=name)
    provenance = _validate_provenance(mapping, context=name)
    epoch = mapping.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"{name}.epoch must be a nonnegative integer")
    if name in _JSONL_STEP_FILES:
        for field in ("microbatch_step", "optimizer_step"):
            value = mapping.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name}.{field} must be a nonnegative integer")
    if name == "runtime_steps.jsonl":
        _require_fields(
            mapping,
            ("candidate", "data_time", "dino_time", "step_time", "allocated_gb", "reserved_gb", "dino_call_count", "mechanism_flags", "samples_per_sec"),
            context=name,
        )
        if not isinstance(mapping["candidate"], str) or not mapping["candidate"]:
            raise ValueError("runtime_steps.jsonl.candidate is required")
        for field in ("data_time", "dino_time", "step_time", "allocated_gb", "reserved_gb", "samples_per_sec"):
            _finite_scalar(mapping[field], context=f"runtime_steps.jsonl.{field}")
        if isinstance(mapping["dino_call_count"], bool) or not isinstance(mapping["dino_call_count"], int):
            raise ValueError("runtime_steps.jsonl.dino_call_count must be an integer")
        flags = _require_mapping(mapping["mechanism_flags"], context="runtime_steps.jsonl.mechanism_flags")
        _require_fields(flags, ("ledger", "pairwise", "counterfactual"), context="runtime_steps.jsonl.mechanism_flags")
        if any(not isinstance(flags[field], bool) for field in ("ledger", "pairwise", "counterfactual")):
            raise ValueError("runtime_steps.jsonl.mechanism_flags must be boolean")
    elif name == "loss_components.jsonl":
        scalar_fields = (
            "action", "reason", "grounding", "pairwise_auxiliary", "counterfactual", "non_regression", "feature_view", "pu_private",
            "grounding_weighted", "pairwise_auxiliary_weighted", "counterfactual_weighted", "non_regression_weighted", "feature_view_weighted",
            "r5", "r10", "total",
        )
        _require_fields(
            mapping,
            (*scalar_fields, "total_optimizer_updates", "valid_counts"),
            context=name,
        )
        for field in scalar_fields:
            _finite_scalar(mapping[field], context=f"loss_components.jsonl.{field}")
        total_updates = mapping["total_optimizer_updates"]
        if isinstance(total_updates, bool) or not isinstance(total_updates, int) or total_updates <= 0:
            raise ValueError("loss_components.jsonl.total_optimizer_updates must be positive")
        optimizer_step = mapping["optimizer_step"]
        expected_r5 = min(1.0, max(0.0, optimizer_step / (0.05 * total_updates)))
        expected_r10 = min(1.0, max(0.0, optimizer_step / (0.10 * total_updates)))
        if not math.isclose(mapping["r5"], expected_r5, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("loss_components.jsonl.r5 must equal the recomputed optimizer schedule")
        if not math.isclose(mapping["r10"], expected_r10, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("loss_components.jsonl.r10 must equal the recomputed optimizer schedule")
        valid_counts = _require_mapping(mapping["valid_counts"], context="loss_components.jsonl.valid_counts")
        _require_fields(valid_counts, ("grounding", "counterfactual"), context="loss_components.jsonl.valid_counts")
        for field in ("grounding", "counterfactual"):
            if isinstance(valid_counts[field], bool) or not isinstance(valid_counts[field], int) or valid_counts[field] < 0:
                raise ValueError(f"loss_components.jsonl.valid_counts.{field} must be a nonnegative integer")
    elif name == "gradient_admission.jsonl":
        _require_fields(
            mapping,
            ("raw_norms", "projected_norms", "cosines", "caps", "ema_norms", "registered", "triggered", "removed"),
            context=name,
        )
        for field in ("raw_norms", "projected_norms", "cosines", "caps", "ema_norms"):
            _finite_mapping(mapping[field], context=f"gradient_admission.jsonl.{field}")
        for field in ("registered", "triggered", "removed"):
            if isinstance(mapping[field], bool) or not isinstance(mapping[field], int) or mapping[field] < 0:
                raise ValueError(f"gradient_admission.jsonl.{field} must be a nonnegative integer")
    elif name == "mechanism_stats.jsonl":
        scalar_fields = (
            "data_time", "dino_time", "field_time", "slot_time", "category_time", "relation_time", "backward_time", "optimizer_time",
            "samples_per_sec", "allocated_gb", "reserved_gb", "action_global_loss", "action_final_loss", "reason_global_loss", "reason_final_loss",
            "action_global_logit_rms", "reason_global_logit_rms", "action_unary_rms_over_global", "action_pairwise_rms_over_global",
            "reason_unary_rms_over_global", "reason_pairwise_rms_over_global", "gamma_AS", "gamma_RA", "gamma_unary", "gamma_pairwise",
            "active_entity_count", "background_mass", "latent_mass", "slot_mask_entropy", "slot_pair_iou", "slot_area_mean", "slot_area_std",
            "entity_type_entropy", "traffic_state_entropy", "road_coverage", "named_contribution_ratio", "latent_contribution_ratio",
            "global_contribution_ratio", "layer_entropy", "positive_weight_mean", "negative_weight_mean", "pu_active_label_count",
            "pu_soft_positive_count", "semantic_private_norm_ratio", "action_reason_context_norm",
            "valid_action_target_count", "valid_reason_target_count",
        )
        counterfactual_scalar_fields = (
            "analytic_selected_effect",
            "feature_selected_effect",
            "control_effect",
            "wrong_target_effect",
            "sign_consistency",
        )
        mapping_fields = (
            "action_layer_weights", "reason_layer_weights", "slot_layer_weights", "layer_collapse", "slot_cos_action_reason",
            "slot_cos_action_grounding", "slot_cos_action_cf", "negative_rates", "projection_rates", "admission_rates", "raw_norms",
            "projected_norms", "budget_hit_rates", "ema_norms", "pu_lambda_by_label",
        )
        _require_fields(
            mapping,
            (
                *scalar_fields,
                *counterfactual_scalar_fields,
                *mapping_fields,
                "counterfactual_available",
                "counterfactual_reason",
                "dino_call_count",
                "optimizer_stepped",
                "owner_gradient_norms",
                "owner_parameter_delta",
            ),
            context=name,
        )
        for field in scalar_fields:
            resolved = _finite_scalar(mapping[field], context=f"mechanism_stats.jsonl.{field}")
            if field.endswith("_time") or field in {"samples_per_sec", "allocated_gb", "reserved_gb"}:
                if resolved < 0.0:
                    raise ValueError(f"mechanism_stats.jsonl.{field} must be nonnegative")
        if not isinstance(mapping["counterfactual_available"], bool):
            raise ValueError("mechanism_stats.jsonl.counterfactual_available must be boolean")
        if mapping["counterfactual_reason"] not in {"formal_128_case_audit", "no_eligible_control"}:
            raise ValueError("mechanism_stats.jsonl.counterfactual_reason is not recognized")
        for field in counterfactual_scalar_fields:
            value = mapping[field]
            if mapping["counterfactual_available"]:
                _finite_scalar(value, context=f"mechanism_stats.jsonl.{field}")
            elif value is not None:
                raise ValueError(f"mechanism_stats.jsonl.{field} must be null when counterfactual is unavailable")
        if mapping["counterfactual_available"] != (
            mapping["counterfactual_reason"] == "formal_128_case_audit"
        ):
            raise ValueError("mechanism_stats.jsonl counterfactual availability and reason disagree")
        for field in mapping_fields:
            _finite_mapping(mapping[field], context=f"mechanism_stats.jsonl.{field}")
        if mapping["dino_call_count"] != 1:
            raise ValueError("mechanism_stats.jsonl.dino_call_count must equal exactly one frozen-DINO call")
        if not isinstance(mapping["optimizer_stepped"], bool):
            raise ValueError("mechanism_stats.jsonl.optimizer_stepped must be boolean")
        for field in ("owner_gradient_norms", "owner_parameter_delta"):
            owner_values = _require_mapping(mapping[field], context=f"mechanism_stats.jsonl.{field}")
            if set(owner_values) != _EXPECTED_OPTIMIZER_OWNERS:
                raise ValueError(f"mechanism_stats.jsonl.{field} must cover exactly the ten optimizer owners")
            for owner, value in owner_values.items():
                resolved = _finite_scalar(value, context=f"mechanism_stats.jsonl.{field}.{owner}")
                if resolved < 0.0:
                    raise ValueError(f"mechanism_stats.jsonl.{field}.{owner} must be nonnegative")
        if mapping["producer"] != "fate_oia.engine.train_acpr_rael_oia:mechanism_stats":
            raise ValueError("mechanism_stats.jsonl.producer must be the formal trainer callsite")
        if not any(value > 0.0 for value in mapping["owner_gradient_norms"].values()):
            raise ValueError("mechanism_stats.jsonl.owner_gradient_norms must contain a nonzero signal")
        if mapping["optimizer_stepped"] and not any(
            value > 0.0 for value in mapping["owner_parameter_delta"].values()
        ):
            raise ValueError(
                "mechanism_stats.jsonl.optimizer_stepped requires a nonzero owner_parameter_delta"
            )
        critical_fields = (
            "action_global_loss",
            "action_final_loss",
            "reason_global_loss",
            "reason_final_loss",
            "gamma_AS",
            "gamma_RA",
            "named_contribution_ratio",
            "analytic_selected_effect",
            "feature_selected_effect",
        )
        if not any(mapping[field] != 0.0 for field in critical_fields):
            raise ValueError("mechanism_stats.jsonl critical mechanism statistics must not all be zero")
    elif name == "metrics_summary.jsonl":
        _require_fields(
            mapping,
            ("raw_action", "raw_reason", "raw_joint", "deploy_action", "deploy_reason", "deploy_joint", "best_flags", "is_best"),
            context=name,
        )
        _metric_bundle(mapping["raw_action"], context="metrics_summary.jsonl.raw_action")
        _metric_bundle(mapping["raw_reason"], context="metrics_summary.jsonl.raw_reason")
        _metric_bundle(mapping["deploy_action"], context="metrics_summary.jsonl.deploy_action")
        _metric_bundle(mapping["deploy_reason"], context="metrics_summary.jsonl.deploy_reason")
        _finite_scalar(mapping["raw_joint"], context="metrics_summary.jsonl.raw_joint")
        _finite_scalar(mapping["deploy_joint"], context="metrics_summary.jsonl.deploy_joint")
        best_flags = _require_mapping(mapping["best_flags"], context="metrics_summary.jsonl.best_flags")
        _require_fields(best_flags, ("deploy_joint", "action_mf1"), context="metrics_summary.jsonl.best_flags")
        if any(not isinstance(best_flags[field], bool) for field in ("deploy_joint", "action_mf1")) or not isinstance(mapping["is_best"], bool):
            raise ValueError("metrics_summary.jsonl best flags must be boolean")
    elif name == "pu_audit.jsonl":
        _require_fields(mapping, ("label_id", "positive_count", "baseline_auprc", "pu_auprc", "delta", "lcb95", "lambda", "decision"), context=name)
        label_id = mapping.get("label_id")
        if isinstance(label_id, bool) or not isinstance(label_id, int) or label_id < 0:
            raise ValueError("pu_audit.jsonl.label_id must be a nonnegative integer")
        if isinstance(mapping["positive_count"], bool) or not isinstance(mapping["positive_count"], int) or mapping["positive_count"] < 0:
            raise ValueError("pu_audit.jsonl.positive_count must be a nonnegative integer")
        for field in ("baseline_auprc", "pu_auprc", "delta", "lcb95", "lambda"):
            _finite_scalar(mapping[field], context=f"pu_audit.jsonl.{field}")
        if not isinstance(mapping.get("decision"), str) or not mapping["decision"]:
            raise ValueError("pu_audit.jsonl.decision is required")
    else:
        raise ValueError(f"unsupported run JSONL schema: {name}")
    _json_safe(mapping, context=name)
    return provenance


def _json_bytes(value: Any, *, context: str) -> bytes:
    return (
        json.dumps(_json_safe(value, context=context), allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]], *, context: str) -> bytes:
    return b"".join(_json_bytes(row, context=f"{context}[{index}]") for index, row in enumerate(rows))


def _yaml_bytes(value: Any, *, context: str) -> bytes:
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError as error:
            raise ValueError(f"{context} is invalid YAML") from error
    return yaml.safe_dump(_json_safe(value, context=context), allow_unicode=False, sort_keys=True).encode("utf-8")


def _tensor_bytes(value: Any, *, context: str) -> bytes:
    buffer = io.BytesIO()
    torch.save(_tensor_safe(value, context=context), buffer)
    return buffer.getvalue()


def _artifact_bytes(name: str, value: Any) -> bytes:
    if name.endswith(".json"):
        return _json_bytes(value, context=name)
    if name.endswith(".jsonl"):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be encoded from a row sequence")
        return _jsonl_bytes(value, context=name)
    if name.endswith(".yaml"):
        return _yaml_bytes(value, context=name)
    if name.endswith(".pt"):
        return _tensor_bytes(value, context=name)
    raise ValueError(f"unsupported artifact extension for {name}")


def _atomic_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_staged_file(path: Path, content: bytes) -> None:
    """Write one already-validated file inside an unpublished epoch directory."""

    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _binary_open_flag(),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record(root: Path, path: Path, kind: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "kind": kind,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        **_json_safe(metadata, context=f"record.{path.name}"),
    }


def _metadata_from_provenance(provenance: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": provenance["schema_version"],
        "producer": provenance["producer"],
        "source_fingerprint_sha256": provenance["source_fingerprint_sha256"],
        "config_sha256": provenance["config_sha256"],
        **({"epoch": provenance["epoch"]} if "epoch" in provenance else {}),
        **({"sample_count": provenance["sample_count"]} if "sample_count" in provenance else {}),
        **extra,
    }


def _parse_existing_jsonl(path: Path, *, name: str) -> list[dict[str, Any]]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"existing JSONL {name} cannot be read as UTF-8") from error
    if not content:
        return []
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(content.splitlines(), start=1):
        if not line:
            raise ValueError(f"existing JSONL {name} contains an empty row at {index}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"existing JSONL {name} contains invalid JSON at {index}") from error
        if not isinstance(row, Mapping):
            raise ValueError(f"existing JSONL {name} row {index} is not an object")
        _validate_run_jsonl_row(name, row)
        rows.append(dict(row))
    return rows


def _assert_monotonic_jsonl(name: str, existing: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> None:
    if name in _JSONL_STEP_FILES and existing:
        previous = existing[-1]
        if row["microbatch_step"] <= previous["microbatch_step"]:
            raise ValueError(f"{name} microbatch_step must be strictly monotonic")
    elif name == "metrics_summary.jsonl" and existing:
        if row["epoch"] <= existing[-1]["epoch"]:
            raise ValueError("metrics_summary.jsonl epoch must be strictly monotonic")


def _read_last_jsonl_row(path: Path, *, name: str) -> dict[str, Any] | None:
    """Read and validate only the final durable JSONL row for hot-path ordering."""

    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if end == 0:
                return None
            stream.seek(end - 1)
            if stream.read(1) == b"\n":
                end -= 1
            if end == 0:
                raise ValueError(f"existing JSONL {name} contains an empty row")

            start = 0
            cursor = end
            while cursor > 0:
                cursor -= 1
                stream.seek(cursor)
                if stream.read(1) == b"\n":
                    start = cursor + 1
                    break
            stream.seek(start)
            encoded = stream.read(end - start)
    except OSError as error:
        raise ValueError(f"existing JSONL {name} cannot be read") from error

    if not encoded:
        raise ValueError(f"existing JSONL {name} contains an empty final row")
    try:
        row = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing JSONL {name} contains invalid JSON in its final row") from error
    if not isinstance(row, Mapping):
        raise ValueError(f"existing JSONL {name} final row is not an object")
    _validate_run_jsonl_row(name, row)
    return dict(row)


def _scan_jsonl_append_state(path: Path) -> tuple[int, int, Any]:
    """Hydrate one reopened writer's append digest/count cache without parsing rows."""

    digest = hashlib.sha256()
    byte_count = 0
    newline_count = 0
    final_byte = b""
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
                newline_count += chunk.count(b"\n")
                final_byte = chunk[-1:]
    except OSError as error:
        raise ValueError(f"existing JSONL {path.name} cannot be read") from error
    if byte_count and final_byte != b"\n":
        newline_count += 1
    return newline_count, byte_count, digest


def _append_jsonl_line(path: Path, content: bytes) -> None:
    """Durably append exactly one already-validated JSONL line under the caller lock."""

    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | _binary_open_flag(),
        0o600,
    )
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(f"failed to append durable JSONL line to {path.name}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _binary_open_flag() -> int:
    """Resolve an unbuffered binary flag even when a Windows test removes os.O_BINARY."""

    flag = getattr(os, "O_BINARY", None)
    if isinstance(flag, int):
        return flag
    if os.name == "nt":
        import msvcrt

        fallback = getattr(msvcrt, "O_BINARY", None)
        if isinstance(fallback, int):
            return fallback
        # Python's msvcrt module does not expose this CRT constant on all builds.
        return 0x8000
    return 0


@contextmanager
def _exclusive_lock(run_root: Path, name: str) -> Iterator[None]:
    """Combine an in-process mutex with an advisory lock usable by other processes."""

    _reject_symlinked_parent(run_root, context="RAEL run_root")
    if not run_root.exists() or run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("RAEL run_root must be a real directory before acquiring a lock")
    lock_dir = run_root / ".rael_locks"
    if lock_dir.exists() and lock_dir.is_symlink():
        raise ValueError("RAEL lock directory must not be a symlink")
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{name}.lock"
    if lock_path.exists() and lock_path.is_symlink():
        raise ValueError("RAEL lock file must not be a symlink")
    key = str(lock_path.resolve())
    with _THREAD_LOCKS_GUARD:
        mutex = _THREAD_LOCKS.setdefault(key, Lock())
    with mutex:
        descriptor = os.open(
            str(lock_path),
            os.O_RDWR | os.O_CREAT | _binary_open_flag(),
            0o600,
        )
        try:
            if os.path.getsize(lock_path) == 0:
                os.write(descriptor, b"0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                deadline = time.monotonic() + 30.0
                while True:
                    try:
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"timed out acquiring RAEL artifact lock {name}")
                        time.sleep(0.02)
                unlock = lambda: msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
                unlock = lambda: fcntl.flock(descriptor, fcntl.LOCK_UN)
            try:
                yield
            finally:
                unlock()
        finally:
            os.close(descriptor)


@contextmanager
def _identity_target_lock(run_root: Path, target_name: str) -> Iterator[None]:
    """Serialize identity binding with every root/epoch publication target."""

    with _exclusive_lock(run_root, "identity"):
        with _exclusive_lock(run_root, target_name):
            yield


def build_rael_repository_fingerprints(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Call P17's live builder without duplicating or mutating its implementation."""

    from fate_oia.engine.train_acpr_rael_oia import (
        build_rael_repository_fingerprints as p17_build_rael_repository_fingerprints,
    )

    return p17_build_rael_repository_fingerprints(*args, **kwargs)


def _owner_manifest_from_live_optimizer(
    trainer: Any,
    *,
    owners: Mapping[str, list[str]],
    no_decay: set[str],
    owner_learning_rates: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bind persisted owner rows to real model parameter objects and optimizer groups."""

    model = getattr(trainer, "model", None)
    bundle = getattr(trainer, "optimizer_bundle", None)
    optimizer = getattr(bundle, "optimizer", None)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("trainer.model must be a public torch.nn.Module for owner artifact validation")
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, Sequence) or isinstance(param_groups, (str, bytes, bytearray)):
        raise TypeError("trainer.optimizer_bundle.optimizer.param_groups must be a public sequence")
    scheduler = getattr(trainer, "scheduler", None)
    scheduler_lrs = (
        tuple(float(value) for value in scheduler.get_last_lr())
        if callable(getattr(scheduler, "get_last_lr", None))
        else None
    )
    if scheduler_lrs is not None and len(scheduler_lrs) != len(param_groups):
        raise ValueError("scheduler learning-rate groups disagree with optimizer.param_groups")

    named_parameters = dict(model.named_parameters())
    expected_names = {name for owner in _OPTIMIZER_OWNER_ORDER for name in owners[owner]}
    if not expected_names.issubset(named_parameters):
        missing = sorted(expected_names.difference(named_parameters))
        raise ValueError(f"trainer owner parameters are absent from model.named_parameters(): {missing}")
    name_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    observed: dict[str, tuple[float, float, float]] = {}
    for group_index, group in enumerate(param_groups):
        group_mapping = _require_mapping(group, context=f"optimizer.param_groups[{group_index}]")
        _require_fields(group_mapping, ("params", "lr", "weight_decay"), context=f"optimizer.param_groups[{group_index}]")
        current_lr = _finite_scalar(
            group_mapping["lr"],
            context=f"optimizer.param_groups[{group_index}].lr",
        )
        base_lr = _finite_scalar(
            group_mapping.get("initial_lr", current_lr),
            context=f"optimizer.param_groups[{group_index}].initial_lr",
        )
        if (
            scheduler_lrs is not None
            and current_lr != scheduler_lrs[group_index]
        ):
            raise ValueError(
                "optimizer.param_groups current lr disagrees with live scheduler"
            )
        decay = _finite_scalar(
            group_mapping["weight_decay"],
            context=f"optimizer.param_groups[{group_index}].weight_decay",
        )
        if decay not in {0.0, _WEIGHT_DECAY}:
            raise ValueError("optimizer.param_groups weight_decay must be 0 or 0.05")
        parameters = group_mapping["params"]
        if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes, bytearray)):
            raise TypeError(f"optimizer.param_groups[{group_index}].params must be a sequence")
        for parameter in parameters:
            parameter_name = name_by_id.get(id(parameter))
            if parameter_name is None or parameter_name not in expected_names:
                raise ValueError("optimizer.param_groups contains a parameter outside formal owner topology")
            if parameter_name in observed:
                raise ValueError("optimizer.param_groups assigns a formal parameter more than once")
            observed[parameter_name] = (base_lr, current_lr, decay)
    if set(observed) != expected_names:
        missing = sorted(expected_names.difference(observed))
        raise ValueError(f"optimizer.param_groups does not cover every formal owner parameter: {missing}")

    manifest: dict[str, dict[str, Any]] = {}
    for owner in _OPTIMIZER_OWNER_ORDER:
        expected_lr = _PRIVATE_OWNER_LR if owner in _PRIVATE_OPTIMIZER_OWNERS else _MAIN_OWNER_LR
        configured_lr = _finite_scalar(
            owner_learning_rates[owner],
            context=f"trainer_config.owner_learning_rates.{owner}",
        )
        if configured_lr != expected_lr:
            raise ValueError(f"trainer owner {owner} violates the fixed public LR contract")
        parameter_names = owners[owner]
        parameters: list[dict[str, Any]] = []
        for parameter_name in parameter_names:
            base_lr, _current_lr, decay = observed[parameter_name]
            expected_decay = 0.0 if parameter_name in no_decay else _WEIGHT_DECAY
            if base_lr != expected_lr or decay != expected_decay:
                raise ValueError(
                    f"optimizer.param_groups disagrees with formal owner/lr/decay for {parameter_name}"
                )
            parameters.append(
                {
                    "name": parameter_name,
                    "lr": base_lr,
                    "weight_decay": decay,
                }
            )
        manifest[owner] = {
            "parameter_names": parameter_names,
            "parameters": parameters,
            "lr": expected_lr,
            "weight_decay": _WEIGHT_DECAY,
            "count": len(parameter_names),
        }
    return manifest


def trainer_run_artifact_contract(
    trainer: Any,
    *,
    artifact_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract a strict, public-only P17 resume contract without private state."""

    context = _require_mapping(artifact_context, context="artifact_context")
    provenance = _validate_provenance(context, context="artifact_context")
    state_dict = getattr(trainer, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("trainer must expose a callable public state_dict()")
    state = _require_mapping(state_dict(), context="trainer state_dict()")
    required = {"checkpoint_schema", "owner_parameter_names", "trainer_config", "resume_fingerprints"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"trainer public state is missing artifact fields: {missing}")
    if not isinstance(state["checkpoint_schema"], str) or not state["checkpoint_schema"]:
        raise ValueError("trainer checkpoint_schema must be a nonempty string")

    owners_raw = _require_mapping(state["owner_parameter_names"], context="trainer owner_parameter_names")
    owners: dict[str, list[str]] = {}
    seen_parameters: set[str] = set()
    for owner, names in owners_raw.items():
        if not isinstance(owner, str) or not owner:
            raise TypeError("trainer owner names must be nonempty strings")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes, bytearray)):
            raise TypeError("trainer owner topology must map owners to parameter-name sequences")
        values = list(names)
        if not values or any(not isinstance(name, str) or not name for name in values):
            raise ValueError("trainer owner parameter names must be nonempty text")
        if len(values) != len(set(values)) or seen_parameters.intersection(values):
            raise ValueError("trainer owner parameter names must be globally unique")
        seen_parameters.update(values)
        owners[owner] = values
    if set(owners) != _EXPECTED_OPTIMIZER_OWNERS:
        raise ValueError(
            "trainer owner topology must contain exactly the ten formal RAEL owners"
        )

    fingerprint_raw = _require_mapping(state["resume_fingerprints"], context="trainer resume_fingerprints")
    source_fingerprint = _validate_p17_v4_fingerprint_manifest(
        {field: fingerprint_raw[field] for field in _FINGERPRINT_FIELDS if field in fingerprint_raw}
    )
    live_fingerprint = _validate_p17_v4_fingerprint_manifest(
        _require_mapping(
            build_rael_repository_fingerprints(phase=source_fingerprint["phase"]),
            context="live P17 fingerprint",
        )
    )
    if source_fingerprint != live_fingerprint:
        raise ValueError("trainer resume fingerprint does not match live P17 builder recomputation")

    trainer_config_raw = _require_mapping(state["trainer_config"], context="trainer trainer_config")
    config_missing = sorted(set(_TRAINER_CONFIG_REQUIRED).difference(trainer_config_raw))
    if config_missing:
        raise ValueError(f"trainer_config is missing fields: {config_missing}")
    trainer_config = _json_safe(
        {field: trainer_config_raw[field] for field in _TRAINER_CONFIG_FIELDS if field in trainer_config_raw},
        context="trainer_config",
    )
    owner_learning_rates = _require_mapping(
        trainer_config["owner_learning_rates"],
        context="trainer_config.owner_learning_rates",
    )
    if set(owner_learning_rates) != _EXPECTED_OPTIMIZER_OWNERS:
        raise ValueError(
            "trainer owner topology and owner_learning_rates must contain the same ten owners"
        )
    bundle = getattr(trainer, "optimizer_bundle", None)
    no_decay_names = getattr(bundle, "no_decay_parameter_names", None)
    if not isinstance(no_decay_names, Sequence) or isinstance(no_decay_names, (str, bytes, bytearray)):
        raise TypeError("trainer.optimizer_bundle.no_decay_parameter_names must be a public parameter-name sequence")
    no_decay = set(no_decay_names)
    if any(not isinstance(name, str) or not name for name in no_decay):
        raise ValueError("trainer.optimizer_bundle.no_decay_parameter_names must contain nonempty text")
    if not no_decay.issubset(seen_parameters):
        raise ValueError("trainer.optimizer_bundle.no_decay_parameter_names must belong to formal owners")

    owner_manifest = _owner_manifest_from_live_optimizer(
        trainer,
        owners=owners,
        no_decay=no_decay,
        owner_learning_rates=owner_learning_rates,
    )
    source_payload = {
        **_metadata_from_provenance(provenance),
        **source_fingerprint,
    }
    owner_payload = {
        **_metadata_from_provenance(provenance),
        "checkpoint_schema": state["checkpoint_schema"],
        "trainer_config": trainer_config,
        "owners": owner_manifest,
    }
    _validate_run_json("source_fingerprint.json", source_payload)
    _validate_run_json("optimizer_owners.json", owner_payload)
    return {
        "source_fingerprint": source_payload,
        "optimizer_owners": owner_payload,
    }


def step_result_artifact_rows(
    step_result: Any,
    *,
    artifact_context: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Convert the public RAELStepResult into appendable, finite artifact rows."""

    required = (
        "components",
        "optimizer_stepped",
        "optimizer_step",
        "microbatch_step",
        "owner_gradient_norms_pre_clip",
        "owner_gradient_norms_post_clip",
        "owner_task_gradient_norms_pre_clip",
        "owner_parameter_delta",
        "owner_optimizer_effect_delta",
        "owner_decay_only_parameter_delta",
        "owner_optimizer_step_count",
        "admission_registered_count",
        "admission_triggered_count",
        "admission_removed_count",
    )
    missing = [name for name in required if not hasattr(step_result, name)]
    if missing:
        raise ValueError(f"public RAELStepResult is missing artifact fields: {missing}")
    components = _finite_float_mapping(step_result.components, context="loss_components")
    common = {
        "optimizer_stepped": bool(step_result.optimizer_stepped),
        "optimizer_step": int(step_result.optimizer_step),
        "microbatch_step": int(step_result.microbatch_step),
    }
    loss_row = {**common, **components}
    gradient_row = {
        **common,
        "owner_gradient_norms_pre_clip": _finite_float_mapping(step_result.owner_gradient_norms_pre_clip, context="owner_gradient_norms_pre_clip"),
        "owner_gradient_norms_post_clip": _finite_float_mapping(step_result.owner_gradient_norms_post_clip, context="owner_gradient_norms_post_clip"),
        "owner_task_gradient_norms_pre_clip": _finite_float_mapping(step_result.owner_task_gradient_norms_pre_clip, context="owner_task_gradient_norms_pre_clip"),
        "owner_parameter_delta": _finite_float_mapping(step_result.owner_parameter_delta, context="owner_parameter_delta"),
        "owner_optimizer_effect_delta": _finite_float_mapping(step_result.owner_optimizer_effect_delta, context="owner_optimizer_effect_delta"),
        "owner_decay_only_parameter_delta": _finite_float_mapping(step_result.owner_decay_only_parameter_delta, context="owner_decay_only_parameter_delta"),
        "owner_optimizer_step_count": {str(name): int(value) for name, value in step_result.owner_optimizer_step_count.items()},
        "registered": int(step_result.admission_registered_count),
        "triggered": int(step_result.admission_triggered_count),
        "removed": int(step_result.admission_removed_count),
    }
    context = _require_mapping(artifact_context, context="artifact_context")
    provenance = _validate_provenance(context, context="artifact_context")
    epoch = context.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("artifact_context.epoch must be a nonnegative integer")
    updates = context.get("total_optimizer_updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("artifact_context.total_optimizer_updates must be positive")
    if "r5" in context or "r10" in context or "r5" in components or "r10" in components:
        raise ValueError("r5/r10 must be derived only from optimizer_step and total_optimizer_updates")
    valid_counts = _require_mapping(context.get("valid_counts"), context="artifact_context.valid_counts")
    _require_fields(valid_counts, ("grounding", "counterfactual"), context="artifact_context.valid_counts")
    for field in ("grounding", "counterfactual"):
        if isinstance(valid_counts[field], bool) or not isinstance(valid_counts[field], int) or valid_counts[field] < 0:
            raise ValueError(f"artifact_context.valid_counts.{field} must be a nonnegative integer")
    admission = _require_mapping(context.get("admission"), context="artifact_context.admission")
    _require_fields(admission, ("raw_norms", "projected_norms", "cosines", "caps", "ema_norms"), context="artifact_context.admission")
    formal_components = (
        "action", "reason", "grounding", "pairwise_auxiliary", "counterfactual", "non_regression", "feature_view", "pu_private",
        "grounding_weighted", "pairwise_auxiliary_weighted", "counterfactual_weighted", "non_regression_weighted",
        "feature_view_weighted", "total",
    )
    _require_fields(components, formal_components, context="RAELStepResult.components")
    for field in formal_components:
        _finite_scalar(components[field], context=f"RAELStepResult.components.{field}")
    r5 = min(1.0, max(0.0, common["optimizer_step"] / (0.05 * updates)))
    r10 = min(1.0, max(0.0, common["optimizer_step"] / (0.10 * updates)))
    context_common = {
        **_metadata_from_provenance(provenance),
        "epoch": epoch,
        "total_optimizer_updates": updates,
    }
    loss_row = {
        **context_common,
        **loss_row,
        "r5": r5,
        "r10": r10,
        "valid_counts": dict(valid_counts),
    }
    gradient_row = {
        **context_common,
        **common,
        "raw_norms": _finite_float_mapping(admission["raw_norms"], context="artifact_context.admission.raw_norms"),
        "projected_norms": _finite_float_mapping(admission["projected_norms"], context="artifact_context.admission.projected_norms"),
        "cosines": _finite_float_mapping(admission["cosines"], context="artifact_context.admission.cosines"),
        "caps": _finite_float_mapping(admission["caps"], context="artifact_context.admission.caps"),
        "ema_norms": _finite_float_mapping(admission["ema_norms"], context="artifact_context.admission.ema_norms"),
        "registered": int(step_result.admission_registered_count),
        "triggered": int(step_result.admission_triggered_count),
        "removed": int(step_result.admission_removed_count),
    }
    _validate_run_jsonl_row("loss_components.jsonl", loss_row)
    _validate_run_jsonl_row("gradient_admission.jsonl", gradient_row)
    _json_safe(loss_row, context="loss_components")
    _json_safe(gradient_row, context="gradient_admission")
    return {"loss_components": loss_row, "gradient_admission": gradient_row}


class RAELArtifactWriter:
    """Persist only the RAEL artifact contract with path, schema, and transaction guards."""

    def __init__(self, run_root: str | Path) -> None:
        raw_root = Path(run_root)
        _reject_symlinked_parent(raw_root, context="RAEL run_root")
        if raw_root.exists() and raw_root.is_symlink():
            raise ValueError("RAEL run_root must not be a symlink")
        self.run_root = raw_root.resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        if self.run_root.is_symlink():
            raise ValueError("RAEL run_root must not resolve through a symlink")
        self._cached_identity: tuple[str, str] | None = None
        self._identity_initialized = False
        self._jsonl_append_states: dict[str, tuple[int, int, Any]] = {}
        self._pu_seen_keys: set[tuple[int, int]] | None = None

    def _revalidate_run_root(self) -> None:
        _reject_symlinked_parent(self.run_root, context="RAEL run_root")
        if not self.run_root.exists() or self.run_root.is_symlink() or not self.run_root.is_dir():
            raise ValueError("RAEL run_root must remain a real non-symlink directory")
        if self.run_root.resolve() != self.run_root:
            raise ValueError("RAEL run_root must not resolve to a different location")

    def _path(self, name: str) -> Path:
        self._revalidate_run_root()
        if Path(name).name != name:
            raise ValueError("RAEL artifact paths must be bare allowed filenames")
        path = self.run_root / name
        if path.exists() and path.is_symlink():
            raise ValueError(f"RAEL artifact path must not be a symlink: {name}")
        return path

    def _epoch_root(self, epoch: int) -> Path:
        self._revalidate_run_root()
        path = self.run_root / f"epoch_{epoch:03d}"
        if path.exists() and path.is_symlink():
            raise ValueError(f"RAEL epoch root must not be a symlink: {path.name}")
        return path

    @staticmethod
    def _provenance_identity(provenance: Mapping[str, Any]) -> tuple[str, str]:
        return (
            str(provenance["source_fingerprint_sha256"]),
            str(provenance["config_sha256"]),
        )

    @classmethod
    def _assert_same_provenance(
        cls,
        values: Sequence[Mapping[str, Any]],
        *,
        context: str,
    ) -> tuple[str, str] | None:
        identities = {cls._provenance_identity(value) for value in values}
        if len(identities) > 1:
            raise ValueError(f"{context} must use one source/config provenance identity")
        return next(iter(identities), None)

    def _read_root_provenances_locked(self, *, require_complete: bool) -> list[dict[str, Any]]:
        """Read every durable root row so a reopened writer cannot bypass provenance binding."""

        provenances: list[dict[str, Any]] = []
        for name in RUN_ROOT_FILES:
            path = self._path(name)
            if not path.exists():
                if require_complete:
                    raise FileNotFoundError(f"RAEL run root is incomplete; missing {name}")
                continue
            if not path.is_file():
                raise ValueError(f"RAEL run root artifact must be a regular file: {name}")
            if name.endswith(".jsonl"):
                rows = _parse_existing_jsonl(path, name=name)
                if require_complete and not rows:
                    raise ValueError(f"RAEL run root JSONL is empty: {name}")
                for row in rows:
                    provenances.append(_validate_run_jsonl_row(name, row))
            elif name.endswith(".yaml"):
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                provenances.append(_validate_run_json(name, payload))
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                provenances.append(_validate_run_json(name, payload))
        self._assert_same_provenance(provenances, context="RAEL run root")
        return provenances

    def _read_immutable_identity_locked(self) -> tuple[str, str] | None:
        """Prefer the two small immutable identity artifacts over a root-wide scan."""

        provenances: list[dict[str, Any]] = []
        for name in ("source_fingerprint.json", "config_resolved.yaml"):
            path = self._path(name)
            if not path.exists():
                continue
            if not path.is_file():
                raise ValueError(f"RAEL immutable identity artifact must be a regular file: {name}")
            try:
                if name.endswith(".yaml"):
                    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                else:
                    payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as error:
                raise ValueError(f"RAEL immutable identity artifact cannot be parsed: {name}") from error
            provenances.append(_validate_run_json(name, payload))
        return self._assert_same_provenance(provenances, context="RAEL immutable identity")

    def _cached_or_discover_identity_locked(self) -> tuple[str, str] | None:
        """Discover one identity once per writer, under the run-wide identity lock."""

        if self._identity_initialized:
            return self._cached_identity
        immutable_identity = self._read_immutable_identity_locked()
        if immutable_identity is None:
            root_provenances = self._read_root_provenances_locked(require_complete=False)
            immutable_identity = self._assert_same_provenance(
                root_provenances,
                context="RAEL run root",
            )
        self._cached_identity = immutable_identity
        self._identity_initialized = True
        return immutable_identity

    def _assert_candidate_matches_root_locked(self, provenance: Mapping[str, Any]) -> None:
        existing_identity = self._cached_or_discover_identity_locked()
        candidate_identity = self._provenance_identity(provenance)
        if existing_identity is not None and existing_identity != candidate_identity:
            raise ValueError("RAEL run root provenance does not match candidate source/config identity")

    def _commit_identity_locked(self, provenance: Mapping[str, Any]) -> None:
        """Cache an identity only after its target is durably published."""

        self._cached_identity = self._provenance_identity(provenance)
        self._identity_initialized = True

    def _jsonl_append_state_locked(self, name: str, path: Path) -> tuple[int, int, Any]:
        """Return cached append accounting, hydrating a reopened writer once if needed."""

        cached = self._jsonl_append_states.get(name)
        if cached is not None:
            row_count, byte_count, digest = cached
            if not path.exists() or path.stat().st_size != byte_count:
                raise ValueError(f"RAEL JSONL append state is stale for {name}")
            return row_count, byte_count, digest
        if not path.exists():
            state = (0, 0, hashlib.sha256())
        else:
            state = _scan_jsonl_append_state(path)
        self._jsonl_append_states[name] = state
        return state

    def _pu_seen_keys_locked(self, path: Path) -> set[tuple[int, int]]:
        """Hydrate the small PU audit index once so uniqueness never depends on a tail row."""

        if self._pu_seen_keys is not None:
            return self._pu_seen_keys
        if not path.exists():
            self._pu_seen_keys = set()
            return self._pu_seen_keys
        rows = _parse_existing_jsonl(path, name="pu_audit.jsonl")
        seen: set[tuple[int, int]] = set()
        for existing in rows:
            key = (int(existing["epoch"]), int(existing["label_id"]))
            if key in seen:
                raise ValueError("pu_audit.jsonl epoch/label_id rows must be unique")
            seen.add(key)
        self._pu_seen_keys = seen
        return seen

    def _jsonl_append_record(
        self,
        *,
        path: Path,
        provenance: Mapping[str, Any],
        row_count: int,
        byte_count: int,
        digest: Any,
    ) -> dict[str, Any]:
        """Build a complete record from incremental state without rereading JSONL history."""

        if path.stat().st_size != byte_count:
            raise ValueError(f"RAEL JSONL append byte accounting is inconsistent for {path.name}")
        return {
            "relative_path": path.relative_to(self.run_root).as_posix(),
            "kind": "run_jsonl",
            "bytes": byte_count,
            "sha256": digest.hexdigest(),
            **_metadata_from_provenance(provenance, row_count=row_count),
        }

    @staticmethod
    def _assert_nonconstant_mechanism_trace(rows: Sequence[Mapping[str, Any]]) -> None:
        """Reject three identical mechanism observations while allowing a first stable row."""

        if len(rows) < 3:
            return
        fields = (
            "dino_call_count",
            "optimizer_stepped",
            "owner_gradient_norms",
            "owner_parameter_delta",
            "action_global_loss",
            "action_final_loss",
            "reason_global_loss",
            "reason_final_loss",
            "gamma_AS",
            "gamma_RA",
            "named_contribution_ratio",
            "analytic_selected_effect",
            "feature_selected_effect",
        )
        signatures = [
            _json_safe({field: row[field] for field in fields}, context="mechanism trace")
            for row in rows[-3:]
        ]
        if signatures[0] == signatures[1] == signatures[2]:
            raise ValueError("mechanism trace contains three consecutive constant observations")

    def write_run_file(self, name: str, payload: Any) -> list[dict[str, Any]]:
        if name not in _RUN_ROOT_SET or name.endswith(".jsonl"):
            raise ValueError(f"run artifact name is not allowed for immutable write: {name}")
        provenance = _validate_run_json(name, payload)
        content = _artifact_bytes(name, payload)
        self._revalidate_run_root()
        with _identity_target_lock(self.run_root, f"immutable-{name}"):
            self._revalidate_run_root()
            self._assert_candidate_matches_root_locked(provenance)
            path = self._path(name)
            if path.exists():
                if path.read_bytes() != content:
                    raise FileExistsError(f"RAEL artifact is immutable and already differs: {path}")
            else:
                _atomic_replace(path, content)
            self._commit_identity_locked(provenance)
            return [_record(self.run_root, path, "run_root", _metadata_from_provenance(provenance))]

    def append_run_jsonl(self, name: str, row: Mapping[str, Any]) -> list[dict[str, Any]]:
        if name not in _RUN_ROOT_SET or not name.endswith(".jsonl"):
            raise ValueError(f"run JSONL artifact name is not allowed: {name}")
        provenance = _validate_run_jsonl_row(name, row)
        encoded_row = _jsonl_bytes([row], context=name)
        self._revalidate_run_root()
        with _identity_target_lock(self.run_root, f"jsonl-{name}"):
            self._revalidate_run_root()
            self._assert_candidate_matches_root_locked(provenance)
            path = self._path(name)
            if path.exists() and path.is_symlink():
                raise ValueError(f"RAEL JSONL artifact must not be a symlink: {name}")
            row_count, byte_count, digest = self._jsonl_append_state_locked(name, path)
            if name == "pu_audit.jsonl":
                pu_seen_keys = self._pu_seen_keys_locked(path)
                pu_key = (int(row["epoch"]), int(row["label_id"]))
                if pu_key in pu_seen_keys:
                    raise ValueError("pu_audit.jsonl epoch/label_id rows must be unique")
            else:
                previous = _read_last_jsonl_row(path, name=name) if row_count else None
                _assert_monotonic_jsonl(name, () if previous is None else (previous,), row)
            separator = b""
            if byte_count:
                with path.open("rb") as stream:
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) != b"\n":
                        separator = b"\n"
            appended = separator + encoded_row
            _append_jsonl_line(path, appended)
            digest.update(appended)
            next_state = (row_count + 1, byte_count + len(appended), digest)
            self._jsonl_append_states[name] = next_state
            if name == "pu_audit.jsonl":
                pu_seen_keys.add(pu_key)
            self._commit_identity_locked(provenance)
            return [
                self._jsonl_append_record(
                    path=path,
                    provenance=provenance,
                    row_count=next_state[0],
                    byte_count=next_state[1],
                    digest=next_state[2],
                )
            ]

    def write_epoch(self, epoch: int, artifacts: Mapping[str, Any]) -> list[dict[str, Any]]:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0 or epoch > 999:
            raise ValueError("epoch must be an integer in [0,999]")
        if not isinstance(artifacts, Mapping) or set(artifacts) != _EPOCH_SET:
            supplied = artifacts if isinstance(artifacts, Mapping) else ()
            missing = sorted(_EPOCH_SET.difference(supplied))
            extra = sorted(set(supplied).difference(_EPOCH_SET))
            raise ValueError(f"epoch artifacts must contain exactly the P18 file set; missing={missing}, extra={extra}")

        self._revalidate_run_root()
        metadata: dict[str, dict[str, Any]] = {}
        for name in EPOCH_FILES:
            if name in {"logits_raw.pt", "logits_deploy.pt"}:
                metadata[name] = _validate_logits_tensor(name, artifacts[name], epoch=epoch)
            elif name == "labels.pt":
                metadata[name] = _validate_labels_tensor(artifacts[name], epoch=epoch)
            elif name.endswith(".jsonl"):
                metadata[name] = _validate_epoch_jsonl(name, artifacts[name], epoch=epoch)
            else:
                metadata[name] = _validate_epoch_json(name, artifacts[name], epoch=epoch)
        epoch_identity = self._assert_same_provenance(
            [metadata[name] for name in EPOCH_FILES],
            context=f"RAEL epoch {epoch:03d}",
        )
        if epoch_identity is None:
            raise AssertionError("complete epoch validation must establish provenance")

        counts = {
            metadata[name]["sample_count"]
            for name in ("logits_raw.pt", "logits_deploy.pt", "labels.pt")
        }
        if len(counts) != 1:
            raise ValueError("epoch tensor artifacts must share one sample_count")
        sample_count = counts.pop()
        if any(metadata[name].get("sample_count") != sample_count for name in EPOCH_FILES):
            raise ValueError("all epoch artifacts must share the tensor sample_count")
        encoded = {name: _artifact_bytes(name, artifacts[name]) for name in EPOCH_FILES}
        epoch_root = self._epoch_root(epoch)

        with _identity_target_lock(self.run_root, f"epoch-{epoch:03d}"):
            self._revalidate_run_root()
            self._assert_candidate_matches_root_locked(metadata[EPOCH_FILES[0]])
            if epoch_root.exists():
                if not epoch_root.is_dir() or epoch_root.is_symlink():
                    raise ValueError("RAEL epoch root must be a real directory")
                existing_files = {path.name for path in epoch_root.iterdir()}
                if existing_files != _EPOCH_SET:
                    raise FileExistsError("RAEL epoch artifacts are immutable and incomplete or polluted")
                for name, content in encoded.items():
                    path = epoch_root / name
                    if path.is_symlink():
                        raise ValueError(f"RAEL epoch artifact must not be a symlink: {name}")
                    if path.read_bytes() != content:
                        raise FileExistsError(f"RAEL epoch artifacts are immutable and already differ: {path}")
            else:
                staging = self.run_root / f".epoch_{epoch:03d}.staging-{uuid.uuid4().hex}"
                try:
                    staging.mkdir(parents=False, exist_ok=False)
                    for name in EPOCH_FILES:
                        _write_staged_file(staging / name, encoded[name])
                    _fsync_directory(staging)
                    os.replace(staging, epoch_root)
                    _fsync_directory(self.run_root)
                except Exception:
                    if staging.exists():
                        shutil.rmtree(staging, ignore_errors=True)
                    raise
            self._commit_identity_locked(metadata[EPOCH_FILES[0]])

        records: list[dict[str, Any]] = []
        for name in EPOCH_FILES:
            extra = {"tensor_shapes": {}} if name.endswith(".pt") else {}
            if name in {"logits_raw.pt", "logits_deploy.pt"}:
                extra["tensor_shapes"] = {"action": [sample_count, 4], "reason": [sample_count, 21]}
            elif name == "labels.pt":
                extra["tensor_shapes"] = {"action": [sample_count, 4], "reason": [sample_count, 21]}
            if name.endswith(".jsonl"):
                extra["row_count"] = metadata[name]["row_count"]
            record_metadata = _metadata_from_provenance(metadata[name], **extra)
            records.append(_record(self.run_root, epoch_root / name, "epoch", record_metadata))
        return records

    def validate_run_root_complete(self) -> list[dict[str, Any]]:
        """Validate one consistent complete-root snapshot under the identity lock."""

        with _identity_target_lock(self.run_root, "complete-validation"):
            return self._validate_run_root_complete_locked()

    def _validate_run_root_complete_locked(self) -> list[dict[str, Any]]:
        """Validate that all twelve root artifacts exist without following symlinks."""

        self._revalidate_run_root()
        root_provenances = self._read_root_provenances_locked(require_complete=True)
        self._assert_same_provenance(root_provenances, context="RAEL run root")
        mechanism_path = self._path("mechanism_stats.jsonl")
        self._assert_nonconstant_mechanism_trace(
            _parse_existing_jsonl(mechanism_path, name="mechanism_stats.jsonl")
        )
        records: list[dict[str, Any]] = []
        for name in RUN_ROOT_FILES:
            path = self._path(name)
            if not path.is_file():
                raise FileNotFoundError(f"RAEL run root is incomplete; missing {name}")
            if name.endswith(".jsonl"):
                rows = _parse_existing_jsonl(path, name=name)
                if not rows:
                    raise ValueError(f"RAEL run root JSONL is empty: {name}")
                provenance = _validate_run_jsonl_row(name, rows[-1])
                records.append(_record(self.run_root, path, "run_jsonl", _metadata_from_provenance(provenance, row_count=len(rows))))
            elif name.endswith(".yaml"):
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
                provenance = _validate_run_json(name, payload)
                records.append(_record(self.run_root, path, "run_root", _metadata_from_provenance(provenance)))
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                provenance = _validate_run_json(name, payload)
                records.append(_record(self.run_root, path, "run_root", _metadata_from_provenance(provenance)))
        return records


__all__ = [
    "EPOCH_FILES",
    "RAELArtifactWriter",
    "RUN_ROOT_FILES",
    "step_result_artifact_rows",
    "trainer_run_artifact_contract",
]
