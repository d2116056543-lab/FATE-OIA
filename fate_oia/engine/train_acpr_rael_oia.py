"""P17: RAEL's sole optimizer, loss, admission, and resume lifecycle owner.

This module intentionally has no data-loader, evaluator, artifact, profiler, or
CLI responsibilities.  Those owners arrive in P18-P21.  P17 owns only the
formal training update from a prepared batch through one optimizer boundary.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fate_oia.datasets.bdd100k_task_aware_index import RAELGroundingRecord
from fate_oia.datasets.rael_dynamic_reliability import (
    DynamicReliabilityResult,
    build_dynamic_reliability,
)
from fate_oia.datasets.rael_grounding_targets import (
    DynamicGroundingBatch,
    build_dynamic_grounding_batch,
    road_grounding_tensor_targets,
    slot_descriptors_from_predictions,
)
from fate_oia.losses.rael_grounding_losses import (
    build_boundary_style_targets,
    entity_attribute_targets_from_p1,
    entity_attribute_grounding_loss,
    road_grounding_loss_bundle,
)
from fate_oia.losses.rael_counterfactual_losses import run_feature_intervention
from fate_oia.losses.rael_pu_losses import (
    REASON_COUNT,
    build_pu_soft_targets,
    canonicalize_sample_id,
    reason_confidence_weights,
    reason_private_pu_loss,
)
from fate_oia.losses.rael_task_losses import (
    evidence_conditional_loss,
    multilabel_asymmetric_loss,
    multilabel_ranking_loss,
    soft_f1_loss,
    two_way_consistency_loss,
)
from fate_oia.optim.rael_gradient_admission import RAELGradientAdmission
from fate_oia.engine.eval_acpr_rael_oia import evaluate_rael_test_only
from fate_oia.utils.rael_artifacts import RAELArtifactWriter


MAIN_LR = 2.0e-4
PRIVATE_LR = 3.0e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP_NORM = 1.0
COUNTERFACTUAL_EVERY_UPDATES = 8


@dataclass(frozen=True)
class RAELWarmupWeights:
    """The only update-dependent RAEL loss schedule."""

    r5: float
    r10: float
    grounding: float
    pairwise_auxiliary: float
    counterfactual: float
    non_regression: float
    feature_view: float


class RAELWarmupSchedule:
    """Single r5/r10 schedule, measured in completed optimizer updates."""

    def __init__(self, *, total_optimizer_updates: int) -> None:
        if total_optimizer_updates <= 0:
            raise ValueError("total_optimizer_updates must be positive")
        self.total_optimizer_updates = int(total_optimizer_updates)

    def weights(self, optimizer_step: int) -> RAELWarmupWeights:
        if optimizer_step < 0:
            raise ValueError("optimizer_step must be nonnegative")
        r5 = min(max(float(optimizer_step) / (0.05 * self.total_optimizer_updates), 0.0), 1.0)
        r10 = min(max(float(optimizer_step) / (0.10 * self.total_optimizer_updates), 0.0), 1.0)
        return RAELWarmupWeights(
            r5=r5,
            r10=r10,
            grounding=0.05 + 0.10 * r5,
            pairwise_auxiliary=0.05 * r10,
            counterfactual=0.05 * r10,
            non_regression=0.02 + 0.03 * r5,
            feature_view=0.02 * r5,
        )


@dataclass
class OwnerOptimizerBundle:
    optimizer: torch.optim.AdamW
    owner_parameter_names: dict[str, tuple[str, ...]]
    owner_learning_rates: dict[str, float]
    no_decay_parameter_names: tuple[str, ...]


@dataclass
class RAELLossBundle:
    action: Tensor
    reason: Tensor
    grounding: Tensor
    pairwise_auxiliary: Tensor
    counterfactual: Tensor
    non_regression: Tensor
    feature_view: Tensor
    pu_private: Tensor
    total: Tensor
    weights: RAELWarmupWeights
    components: dict[str, Tensor]
    pu_soft_targets: Tensor


@dataclass
class RAELStepResult:
    components: dict[str, Tensor]
    optimizer_stepped: bool
    optimizer_step: int
    microbatch_step: int
    admission_hook_count: int
    owner_gradient_norms_pre_clip: dict[str, float]
    owner_gradient_norms_post_clip: dict[str, float]
    owner_task_gradient_norms_pre_clip: dict[str, float]
    owner_parameter_delta: dict[str, float]
    owner_optimizer_effect_delta: dict[str, float]
    owner_decay_only_parameter_delta: dict[str, float]
    owner_optimizer_step_count: dict[str, int]
    # ``admission_hook_count`` is retained for callers of the original P17
    # result contract.  It is now the real number of hooks that fired.
    admission_registered_count: int
    admission_triggered_count: int
    admission_removed_count: int
    mechanism_observation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RAELEncodedFieldHandoff:
    """One-DINO, detached public replay boundary for P20/P21 diagnostics.

    The trainer never mutates ``visual_field`` or ``outputs`` after handing
    them off.  Consumers must use ``replay_counterfactual_from_encoded_field``
    rather than re-encoding the image batch.
    """

    visual_field: Any
    outputs: Mapping[str, Any]
    file_names: tuple[str, ...]
    dino_call_count_before: int | None
    dino_call_count_after: int | None


class ReZeroBootstrapTracker:
    """Enforces the P16 contribution bootstrap deadlines at update 0/1/2."""

    _REQUIRED = (
        "bridge_output",
        "unary_output",
        "pairwise_output",
        "bridge_internal",
        "unary_internal",
        "pairwise_internal",
    )

    def __init__(self) -> None:
        self._records: dict[int, dict[str, float]] = {}

    def observe(self, optimizer_step: int, norms: Mapping[str, float]) -> None:
        if optimizer_step < 0:
            raise ValueError("optimizer_step must be nonnegative")
        missing = [name for name in self._REQUIRED if name not in norms]
        if missing:
            raise ValueError(f"ReZero bootstrap metrics missing: {missing}")
        self._records[int(optimizer_step)] = {name: float(norms[name]) for name in self._REQUIRED}

    def assert_satisfied(self) -> None:
        if 0 in self._records:
            record = self._records[0]
            for name in ("bridge_output", "unary_output"):
                if record[name] <= 0.0:
                    raise RuntimeError(f"ReZero bootstrap failed at update0: {name} gradient is zero")
        if 1 in self._records:
            record = self._records[1]
            for name in ("bridge_internal", "unary_internal", "pairwise_output"):
                if record[name] <= 0.0:
                    raise RuntimeError(f"ReZero bootstrap failed at update1: {name} gradient is zero")
        if 2 in self._records and self._records[2]["pairwise_internal"] <= 0.0:
            raise RuntimeError("ReZero bootstrap failed at update2: pairwise internal gradient is zero")

    def state_dict(self) -> dict[str, Any]:
        return {"records": copy.deepcopy(self._records)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        records = state.get("records", {})
        self._records = {
            int(step): {name: float(value) for name, value in values.items()}
            for step, values in records.items()
        }


def _owner_prefixes() -> dict[str, tuple[str, ...]]:
    return {
        "multilayer_field": ("multilayer_field",),
        "slot_ledger_core": ("slot_ledger",),
        "slot_attribute_heads": ("slot_attribute_heads",),
        "action_category": ("action_category",),
        "semantic_reason": ("semantic_reason",),
        "action_reason_bridge": ("action_reason_bridge",),
        "unary_contribution": ("action_unary", "reason_unary"),
        "pairwise_relation": ("action_pairwise", "reason_pairwise"),
        "reason_private": ("reason_private",),
        "pu_private": ("pu_private_head",),
    }


def _is_calibration_parameter(name: str) -> bool:
    normalized = name.lower()
    return any(token in normalized for token in ("calibration", "threshold", "temperature", "posthoc"))


def _is_no_decay(name: str, parameter: nn.Parameter, model: nn.Module) -> bool:
    if parameter.ndim <= 1 or name.endswith(".bias"):
        return True
    normalized = name.lower()
    if any(token in normalized for token in ("norm", "embedding")):
        return True
    module_path = name.rsplit(".", 1)[0] if "." in name else ""
    try:
        parent = model.get_submodule(module_path) if module_path else model
    except AttributeError:
        parent = model
    return isinstance(parent, (nn.Embedding, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


def build_rael_optimizer(model: nn.Module) -> OwnerOptimizerBundle:
    """Build P17's exact ten-owner AdamW topology.

    Frozen DINO and posthoc calibration parameters are intentionally excluded.
    Every remaining trainable representation parameter must have exactly one
    owner; ambiguity is an error rather than a silent fallback.
    """

    all_named = dict(model.named_parameters())
    owner_parameter_names: dict[str, tuple[str, ...]] = {}
    owner_lrs: dict[str, float] = {}
    consumed: set[str] = set()
    no_decay: list[str] = []
    parameter_groups: list[dict[str, Any]] = []

    for owner, prefixes in _owner_prefixes().items():
        selected = tuple(
            name
            for name, parameter in all_named.items()
            if parameter.requires_grad and any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        )
        if not selected:
            raise ValueError(f"P17 optimizer owner {owner!r} has no trainable parameters")
        overlap = consumed.intersection(selected)
        if overlap:
            raise ValueError(f"P17 parameter owner overlap for {owner}: {sorted(overlap)}")
        consumed.update(selected)
        owner_parameter_names[owner] = selected
        learning_rate = PRIVATE_LR if owner in {"reason_private", "pu_private"} else MAIN_LR
        owner_lrs[owner] = learning_rate
        decay_parameters: list[nn.Parameter] = []
        no_decay_parameters: list[nn.Parameter] = []
        for name in selected:
            parameter = all_named[name]
            if _is_no_decay(name, parameter, model):
                no_decay.append(name)
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        if decay_parameters:
            parameter_groups.append({"params": decay_parameters, "lr": learning_rate, "weight_decay": WEIGHT_DECAY, "owner": owner})
        if no_decay_parameters:
            parameter_groups.append({"params": no_decay_parameters, "lr": learning_rate, "weight_decay": 0.0, "owner": owner})

    unexpected = {
        name
        for name, parameter in all_named.items()
        if parameter.requires_grad
        and name not in consumed
        and not name.startswith("dino_extractor.")
        and not _is_calibration_parameter(name)
    }
    if unexpected:
        raise ValueError(f"P17 has unowned trainable representation parameters: {sorted(unexpected)}")
    return OwnerOptimizerBundle(
        optimizer=torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999), eps=1.0e-8),
        owner_parameter_names=owner_parameter_names,
        owner_learning_rates=owner_lrs,
        no_decay_parameter_names=tuple(sorted(no_decay)),
    )


def _cosine_lambda(total_updates: int, warmup_ratio: float = 0.05) -> Callable[[int], float]:
    warmup_steps = max(1, int(math.ceil(total_updates * warmup_ratio)))

    def value(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(max((step - warmup_steps) / max(1, total_updates - warmup_steps), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return value


def _scalar(value: Tensor | float, *, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _zero(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def _parameter_delta(before: Mapping[str, Tensor], after: Mapping[str, Tensor], names: Sequence[str]) -> float:
    squares = [(after[name].detach().float() - before[name].detach().float()).square().sum() for name in names]
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


def _checkpoint_clone(value: Any) -> Any:
    """Clone mutable checkpoint payloads without retaining model graph/storage."""

    if isinstance(value, Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, Mapping):
        return {str(key): _checkpoint_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_checkpoint_clone(item) for item in value)
    if isinstance(value, list):
        return [_checkpoint_clone(item) for item in value]
    return copy.deepcopy(value)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible immutable resume contract deterministically."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_RAEL_FINGERPRINT_SCHEMA = "rael-repository-fingerprint-v4"
_RAEL_FINGERPRINT_GROUPS = ("source", "test", "config", "schema", "skill", "script")
_RAEL_DECLARED_GROUPS: dict[str, tuple[Path, ...]] = {
    "source": tuple(
        Path(path)
        for path in (
            "fate_oia/datasets/bdd_oia_multitask.py",
            "fate_oia/datasets/bdd100k_task_aware_index.py",
            "fate_oia/datasets/rael_dynamic_reliability.py",
            "fate_oia/datasets/rael_grounding_targets.py",
            "fate_oia/engine/audit_acpr_rael_oia.py",
            "fate_oia/engine/eval_acpr_rael_oia.py",
            "fate_oia/engine/export_rael_cases.py",
            "fate_oia/engine/profile_acpr_rael_oia.py",
            "fate_oia/engine/supervise_acpr_rael_oia_foreground.py",
            "fate_oia/engine/train_acpr_rael_oia.py",
            "fate_oia/losses/rael_counterfactual_losses.py",
            "fate_oia/losses/rael_grounding_losses.py",
            "fate_oia/losses/rael_pu_losses.py",
            "fate_oia/losses/rael_task_losses.py",
            "fate_oia/metrics.py",
            "fate_oia/models/acpr_sparse_ops.py",
            "fate_oia/models/rael_action_reason_bridge.py",
            "fate_oia/models/rael_category_foundation.py",
            "fate_oia/models/rael_dino_field.py",
            "fate_oia/models/rael_multilayer_field.py",
            "fate_oia/models/rael_oia_model.py",
            "fate_oia/models/rael_reason_private.py",
            "fate_oia/models/rael_relation_contributions.py",
            "fate_oia/models/rael_semantic_reason.py",
            "fate_oia/models/rael_slot_ledger.py",
            "fate_oia/optim/rael_gradient_admission.py",
            "fate_oia/threshold_tuning.py",
            "fate_oia/transforms.py",
            "fate_oia/transforms_rael.py",
            "fate_oia/utils/acpr_threshold_search.py",
            "fate_oia/utils/acpr_thresholds.py",
            "fate_oia/utils/acpr_train_calib_split.py",
            "fate_oia/utils/rael_artifacts.py",
            "fate_oia/utils/rael_posthoc_calibration.py",
            "fate_oia/utils/rael_runtime.py",
            "fate_oia/utils/rael_schema.py",
        )
    ),
    "test": tuple(
        Path(path)
        for path in (
            "tests/test_rael_absence_evidence.py",
            "tests/test_rael_action_reason_firewall.py",
            "tests/test_rael_adaptive_entmax.py",
            "tests/test_rael_artifacts.py",
            "tests/test_rael_audit.py",
            "tests/test_rael_category_foundation.py",
            "tests/test_rael_counterfactual.py",
            "tests/test_rael_dino_contract.py",
            "tests/test_rael_dynamic_grounding.py",
            "tests/test_rael_dynamic_reliability.py",
            "tests/test_rael_case_export.py",
            "tests/test_rael_epoch_adapter.py",
            "tests/test_rael_eval_contract.py",
            "tests/test_rael_gradient_admission.py",
            "tests/test_rael_grounding_index.py",
            "tests/test_rael_grounding_layout.py",
            "tests/test_rael_model_forward.py",
            "tests/test_rael_multilayer_reading.py",
            "tests/test_rael_pairwise_relation.py",
            "tests/test_rael_posthoc_calibration.py",
            "tests/test_rael_pu.py",
            "tests/test_rael_reason_private.py",
            "tests/test_rael_reason_schema.py",
            "tests/test_rael_runtime.py",
            "tests/test_rael_semantic_reason.py",
            "tests/test_rael_slot_attributes.py",
            "tests/test_rael_slot_competition.py",
            "tests/test_rael_supervisor.py",
            "tests/test_rael_train_protocol.py",
            "tests/test_rael_trainer_handoff.py",
            "tests/test_rael_unary_contribution.py",
            "tests/test_rael_worktree_contract.py",
        )
    ),
    "config": (Path("configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml"),),
    "schema": (
        Path("configs/rael_action_semantics.yaml"),
        Path("configs/rael_reason_semantics.yaml"),
        Path("configs/rael_slot_schema.yaml"),
    ),
    "skill": (Path(".codex/skills/rael-oia-v1-implementation-audit/SKILL.md"),),
    "script": (Path("scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1"),),
}


def rael_repository_fingerprint_files(
    repository_root: str | Path | None = None,
) -> dict[str, tuple[Path, ...]]:
    """Return declared files plus their repo-local transitive Python closure."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    if set(_RAEL_DECLARED_GROUPS) != set(_RAEL_FINGERPRINT_GROUPS):
        raise RuntimeError("RAEL declared fingerprint group coverage is inconsistent")
    groups = {
        group: tuple(sorted(set(paths), key=lambda path: path.as_posix()))
        for group, paths in _RAEL_DECLARED_GROUPS.items()
    }
    declared = {path for paths in groups.values() for path in paths}
    python_roots = [
        path
        for group in ("source", "test")
        for path in groups[group]
        if path.suffix == ".py" and (root / path).is_file()
    ]
    closure = _repository_python_import_closure(root, python_roots)
    groups["source"] = tuple(
        sorted(
            (*groups["source"], *(path for path in closure if path not in declared)),
            key=lambda path: path.as_posix(),
        )
    )
    return groups


def _normalized_relative_path(path: Path) -> str:
    normalized = path.as_posix()
    parts = normalized.split("/")
    if (
        path.is_absolute()
        or not normalized
        or "\\" in normalized
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"RAEL fingerprint path must be normalized and relative: {path}")
    return normalized


def _sha256_file(root: Path, relative: Path) -> str:
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


def _module_name_for_path(relative: Path) -> tuple[str, bool] | None:
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts.pop()
    if not parts:
        return None
    return ".".join(parts), is_package


def _module_candidates(root: Path, module_name: str) -> tuple[Path, ...]:
    if not module_name or not (
        module_name == "fate_oia" or module_name.startswith("fate_oia.")
    ):
        return ()
    base = Path(*module_name.split("."))
    candidates = (base.with_suffix(".py"), base / "__init__.py")
    return tuple(path for path in candidates if (root / path).is_file())


def _imports_from_python_file(root: Path, relative: Path) -> tuple[Path, ...]:
    module_info = _module_name_for_path(relative)
    if module_info is None:
        return ()
    module_name, is_package = module_info
    package_parts = module_name.split(".") if is_package else module_name.split(".")[:-1]
    try:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"), filename=str(relative))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ()
    resolved: set[Path] = set()
    for node in ast.walk(tree):
        module_names: list[str] = []
        if isinstance(node, ast.Import):
            module_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                if keep < 0:
                    continue
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base_name = ".".join(base_parts)
            else:
                base_name = node.module or ""
            if base_name:
                module_names.append(base_name)
                module_names.extend(
                    f"{base_name}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
        for imported in module_names:
            resolved.update(_module_candidates(root, imported))
    return tuple(sorted(resolved, key=lambda path: path.as_posix()))


def _repository_python_import_closure(
    root: Path,
    roots: Sequence[Path],
) -> tuple[Path, ...]:
    pending = list(roots)
    seen: set[Path] = set()
    while pending:
        relative = pending.pop()
        if relative in seen or not (root / relative).is_file():
            continue
        seen.add(relative)
        pending.extend(
            dependency
            for dependency in _imports_from_python_file(root, relative)
            if dependency not in seen
        )
    return tuple(sorted(seen, key=lambda path: path.as_posix()))


def _manifest_entries_hash(
    *,
    namespace: str,
    phase: str,
    paths: Sequence[str],
    file_status: Mapping[str, str],
    file_sha256: Mapping[str, str | None],
) -> str:
    return _stable_hash(
        {
            "namespace": namespace,
            "schema": _RAEL_FINGERPRINT_SCHEMA,
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


def _validate_rael_fingerprint_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the persisted v4 file-level resume contract."""

    required = {
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
    }
    if set(payload) != required:
        raise ValueError("fingerprint manifest fields are incomplete or unexpected")
    if payload["fingerprint_schema"] != _RAEL_FINGERPRINT_SCHEMA:
        raise ValueError("fingerprint manifest schema mismatch")
    phase = payload["phase"]
    if not isinstance(phase, str) or not phase or phase.strip() != phase:
        raise ValueError("fingerprint manifest phase must be nonempty normalized text")
    if not isinstance(payload["complete"], bool):
        raise ValueError("fingerprint manifest complete must be boolean")

    groups_raw = payload["groups"]
    status_raw = payload["file_status"]
    hashes_raw = payload["file_sha256"]
    group_hashes_raw = payload["group_hashes"]
    if not all(
        isinstance(value, Mapping)
        for value in (groups_raw, status_raw, hashes_raw, group_hashes_raw)
    ):
        raise ValueError("fingerprint manifest groups and hashes must be mappings")
    if set(groups_raw) != set(_RAEL_FINGERPRINT_GROUPS) or set(group_hashes_raw) != set(_RAEL_FINGERPRINT_GROUPS):
        raise ValueError("fingerprint manifest group coverage mismatch")

    normalized_groups: dict[str, list[str]] = {}
    ordered_paths: list[str] = []
    for group in _RAEL_FINGERPRINT_GROUPS:
        paths = groups_raw[group]
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"fingerprint manifest group {group} must be a nonempty list")
        normalized = []
        for relative in paths:
            if not isinstance(relative, str):
                raise ValueError(f"fingerprint manifest path in {group} must be text")
            normalized.append(_normalized_relative_path(Path(relative)))
        if normalized != sorted(normalized):
            raise ValueError(f"fingerprint manifest group {group} is not deterministically sorted")
        normalized_groups[group] = normalized
        ordered_paths.extend(normalized)
    if len(ordered_paths) != len(set(ordered_paths)):
        raise ValueError("fingerprint manifest cannot assign a required file to multiple groups")

    def _valid_digest(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    if set(status_raw) != set(ordered_paths) or set(hashes_raw) != set(ordered_paths):
        raise ValueError("fingerprint manifest per-file coverage is invalid")
    for path in ordered_paths:
        status = status_raw[path]
        digest = hashes_raw[path]
        if status not in {"present", "missing"}:
            raise ValueError("fingerprint manifest file status is invalid")
        if (status == "present" and not _valid_digest(digest)) or (
            status == "missing" and digest is not None
        ):
            raise ValueError("fingerprint manifest status/SHA256 relation is invalid")
    expected_missing = sorted(path for path in ordered_paths if status_raw[path] == "missing")
    if payload["missing_files"] != expected_missing:
        raise ValueError("fingerprint manifest missing-file set is inconsistent")
    if payload["complete"] != (not expected_missing):
        raise ValueError("fingerprint manifest completeness is inconsistent")
    if any(not _valid_digest(group_hashes_raw[group]) for group in _RAEL_FINGERPRINT_GROUPS):
        raise ValueError("fingerprint manifest group SHA256 is invalid")
    aggregate_names = ("source_hash", "config_hash", "schema_hash", "required_files_hash")
    if any(not _valid_digest(payload[name]) for name in aggregate_names):
        raise ValueError("fingerprint manifest aggregate SHA256 is invalid")

    return {
        "fingerprint_schema": _RAEL_FINGERPRINT_SCHEMA,
        "phase": phase,
        "complete": bool(payload["complete"]),
        "groups": normalized_groups,
        "file_status": {
            path: str(status_raw[path])
            for path in sorted(ordered_paths)
        },
        "file_sha256": {
            path: (str(hashes_raw[path]) if hashes_raw[path] is not None else None)
            for path in sorted(ordered_paths)
        },
        "missing_files": expected_missing,
        "group_hashes": {
            group: str(group_hashes_raw[group])
            for group in _RAEL_FINGERPRINT_GROUPS
        },
        **{name: str(payload[name]) for name in aggregate_names},
    }


def build_rael_repository_fingerprints(
    repository_root: str | Path | None = None,
    *,
    phase: str = "development",
) -> dict[str, Any]:
    """Build the artifact/audit-ready per-file RAEL resume fingerprint manifest."""

    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    groups = rael_repository_fingerprint_files(root)
    group_paths = {
        group: [_normalized_relative_path(path) for path in groups[group]]
        for group in _RAEL_FINGERPRINT_GROUPS
    }
    all_paths = tuple(
        path
        for group in _RAEL_FINGERPRINT_GROUPS
        for path in groups[group]
    )
    file_status = {
        _normalized_relative_path(path): (
            "present" if (root / path).is_file() else "missing"
        )
        for path in sorted(all_paths, key=lambda path: path.as_posix())
    }
    file_sha256 = {
        normalized: (_sha256_file(root, Path(normalized)) if status == "present" else None)
        for normalized, status in file_status.items()
    }
    missing_files = sorted(path for path, status in file_status.items() if status == "missing")
    group_hashes = {
        group: _manifest_entries_hash(
            namespace=f"rael-{group}-v4",
            phase=phase,
            paths=group_paths[group],
            file_status=file_status,
            file_sha256=file_sha256,
        )
        for group in _RAEL_FINGERPRINT_GROUPS
    }
    all_normalized = [path for group in _RAEL_FINGERPRINT_GROUPS for path in group_paths[group]]
    manifest: dict[str, Any] = {
        "fingerprint_schema": _RAEL_FINGERPRINT_SCHEMA,
        "phase": phase,
        "complete": not missing_files,
        "groups": group_paths,
        "file_status": file_status,
        "file_sha256": file_sha256,
        "missing_files": missing_files,
        "group_hashes": group_hashes,
        "source_hash": _stable_hash(
            {
                "namespace": "rael-source-test-skill-script-v4",
                "phase": phase,
                "groups": {
                    group: group_hashes[group]
                    for group in ("source", "test", "skill", "script")
                },
            }
        ),
        "config_hash": group_hashes["config"],
        "schema_hash": group_hashes["schema"],
        "required_files_hash": _manifest_entries_hash(
            namespace="rael-required-declared-and-import-closure-v4",
            phase=phase,
            paths=all_normalized,
            file_status=file_status,
            file_sha256=file_sha256,
        ),
    }
    return _validate_rael_fingerprint_manifest(manifest)


def _unique_model_device(model: nn.Module) -> torch.device:
    devices = {
        tensor.device
        for tensor in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if len(devices) != 1:
        raise ValueError(
            "P17 requires all model parameters and buffers on one device before trainer construction"
        )
    return next(iter(devices))


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def _capture_forward_scalar_state(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Capture non-state-dict scalar counters that formal forwards may mutate."""

    captured: dict[str, dict[str, Any]] = {}
    for module_name, module in model.named_modules():
        values = {
            name: copy.deepcopy(value)
            for name, value in vars(module).items()
            if name != "training"
            and isinstance(value, (bool, int, float, str, type(None)))
        }
        if values:
            captured[module_name] = values
    return captured


def _restore_forward_scalar_state(
    model: nn.Module, state: Mapping[str, Mapping[str, Any]]
) -> None:
    modules = dict(model.named_modules())
    for module_name, values in state.items():
        module = modules.get(module_name)
        if module is None:
            raise RuntimeError(f"P17 rollback missing module {module_name!r}")
        for name, value in values.items():
            setattr(module, name, copy.deepcopy(value))


class RAELTrainer:
    """Formal P17 update owner, deliberately independent of data/eval/artifacts."""

    def __init__(
        self,
        model: nn.Module,
        *,
        total_optimizer_updates: int,
        gradient_accumulation_steps: int,
        precision: str = "bf16",
        counterfactual_loss_fn: Callable[[dict[str, Any], int], Tensor] | None = None,
        resume_fingerprints: Mapping[str, Any] | None = None,
    ) -> None:
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if precision != "bf16":
            raise ValueError("formal RAEL P17 precision must be bf16")
        self.model = model
        self.schedule = RAELWarmupSchedule(total_optimizer_updates=total_optimizer_updates)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.precision = precision
        self.optimizer_bundle = build_rael_optimizer(model)
        self.optimizer = self.optimizer_bundle.optimizer
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_cosine_lambda(total_optimizer_updates)
        )
        self.admission = RAELGradientAdmission(
            state_device=_unique_model_device(model)
        )
        self.counterfactual_loss_fn = counterfactual_loss_fn
        self.last_counterfactual_result: dict[str, Any] = {
            "available": False,
            "reason": "not_scheduled",
        }
        self.last_admission_summary: dict[str, dict[str, float]] | None = None
        self.last_epoch_pu_soft_positive: Tensor | None = None
        self._last_pu_soft_targets: Tensor | None = None
        self.last_dynamic_grounding_batch: DynamicGroundingBatch | None = None
        self.last_dynamic_grounding_targets: dict[str, Any] | None = None
        self.last_dynamic_reliability: DynamicReliabilityResult | None = None
        self.microbatch_step = 0
        self.optimizer_step = 0
        self.epoch = 0
        self.pu_lambda = torch.zeros(REASON_COUNT, dtype=torch.float32)
        self.pu_active_labels = torch.zeros(REASON_COUNT, dtype=torch.bool)
        self.owner_optimizer_step_count = {name: 0 for name in self.optimizer_bundle.owner_parameter_names}
        self.bootstrap = ReZeroBootstrapTracker()
        self._view_ema_state: dict[str, Any] = {}
        self._posthoc_calibration_state: dict[str, Any] = {}
        self._resume_fingerprints = self._build_resume_fingerprints(resume_fingerprints)
        self.optimizer.zero_grad(set_to_none=True)

    def _build_resume_fingerprints(self, supplied: Mapping[str, Any] | None) -> dict[str, Any]:
        """Create the immutable P17 v3 source/test/config/schema/skill identity."""

        try:
            current = _validate_rael_fingerprint_manifest(
                build_rael_repository_fingerprints()
            )
        except ValueError as error:
            raise ValueError(
                f"current repository fingerprint manifest is invalid: {error}"
            ) from error
        if supplied is None:
            return current
        try:
            expected = _validate_rael_fingerprint_manifest(supplied)
        except ValueError as error:
            raise ValueError(
                f"resume_fingerprints must be a valid v3 fingerprint manifest: {error}"
            ) from error
        if expected != current:
            raise ValueError(
                "supplied fingerprint manifest does not match current repository"
            )
        return current

    def set_view_ema_state(self, state: Mapping[str, Any]) -> None:
        """Accept the current grounding/view producer EMA for P17 checkpointing."""

        if not isinstance(state, Mapping):
            raise TypeError("view_ema_state must be a mapping")
        self._view_ema_state = _checkpoint_clone(state)

    def set_posthoc_calibration_state(self, state: Mapping[str, Any]) -> None:
        """Accept epoch-end calibration output without optimizing it in P17."""

        if not isinstance(state, Mapping):
            raise TypeError("posthoc_calibration_state must be a mapping")
        self._posthoc_calibration_state = _checkpoint_clone(state)

    def set_pu_label_gate(self, active_labels: Tensor, lambda_by_label: Tensor) -> None:
        if active_labels.dtype != torch.bool or active_labels.shape != (REASON_COUNT,):
            raise ValueError("active_labels must be bool [21]")
        if lambda_by_label.shape != (REASON_COUNT,) or not torch.is_floating_point(lambda_by_label):
            raise ValueError("lambda_by_label must be floating point [21]")
        if not bool(torch.isfinite(lambda_by_label).all()) or not bool(((lambda_by_label >= 0.0) & (lambda_by_label <= 0.20)).all()):
            raise ValueError("PU lambdas must be finite in [0,0.20]")
        self.pu_active_labels = active_labels.detach().cpu().clone()
        self.pu_lambda = (lambda_by_label.detach().float().cpu() * self.pu_active_labels.float()).clone()
        setter = getattr(self.model, "set_pu_active_labels", None)
        if setter is not None:
            setter(active_labels.to(next(self.model.parameters()).device))

    def _set_epoch_pu_state(self, epoch: int) -> tuple[Tensor, Tensor]:
        if epoch == 0:
            active = torch.zeros_like(self.pu_active_labels)
            values = torch.zeros_like(self.pu_lambda)
        else:
            active = self.pu_active_labels
            values = self.pu_lambda * active.float()
        setter = getattr(self.model, "set_pu_active_labels", None)
        if setter is not None:
            setter(active.to(next(self.model.parameters()).device))
        return active, values

    @staticmethod
    def _dynamic_image_sizes(
        batch: Mapping[str, Any], batch_size: int
    ) -> tuple[tuple[int, int], ...]:
        meta = batch.get("transform_meta")
        if not isinstance(meta, (tuple, list)) or len(meta) != batch_size:
            raise ValueError(
                "dynamic grounding requires transform_meta for every sample"
            )
        sizes: list[tuple[int, int]] = []
        for item in meta:
            if not isinstance(item, Mapping):
                raise TypeError("transform_meta must contain mappings")
            size = item.get("image_size")
            if (
                not isinstance(size, (tuple, list))
                or len(size) != 2
                or not all(isinstance(value, (int, float)) for value in size)
            ):
                raise ValueError("transform_meta image_size must be [width,height]")
            sizes.append((int(size[0]), int(size[1])))
        return tuple(sizes)

    @staticmethod
    def _concatenate_target_mappings(
        values: Sequence[Mapping[str, Tensor]],
    ) -> dict[str, Tensor]:
        if not values:
            raise ValueError("cannot concatenate an empty grounding batch")
        names = tuple(values[0])
        if any(tuple(item) != names for item in values):
            raise ValueError("grounding target rows must have identical fields")
        return {name: torch.cat([item[name] for item in values], dim=0) for name in names}

    def _build_dynamic_grounding_targets(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        *,
        reference: Tensor,
    ) -> dict[str, Any]:
        records = batch.get("grounding_records")
        batch_size = int(reference.shape[0])
        if not isinstance(records, (tuple, list)) or len(records) != batch_size:
            raise ValueError(
                "dynamic grounding requires grounding_records for every sample"
            )
        if not all(isinstance(record, RAELGroundingRecord) for record in records):
            raise TypeError("grounding_records must contain transformed RAELGroundingRecord values")
        image_sizes = self._dynamic_image_sizes(batch, batch_size)
        horizontal = outputs.get("slot_sector_probs")
        if not isinstance(horizontal, Mapping):
            raise ValueError("dynamic grounding requires slot_sector_probs")
        required = {
            "slot_centroid": outputs.get("slot_centroid"),
            "slot_scale": outputs.get("slot_scale"),
            "slot_type_probs": outputs.get("slot_type_probs"),
            "horizontal": horizontal.get("horizontal"),
        }
        if not all(isinstance(value, Tensor) for value in required.values()):
            raise ValueError("dynamic grounding requires current slot geometry/type/sector tensors")
        descriptors = slot_descriptors_from_predictions(
            required["slot_centroid"][:, :12],
            required["slot_scale"][:, :12],
            required["slot_type_probs"],
            required["horizontal"][:, :12],
            image_sizes,
        )
        dynamic = build_dynamic_grounding_batch(descriptors, records, image_sizes)
        grounding_outputs = outputs.get("grounding_outputs")
        if not isinstance(grounding_outputs, Mapping) or not isinstance(
            grounding_outputs.get("road"), Mapping
        ):
            raise ValueError("dynamic grounding requires current road outputs")
        road_outputs = grounding_outputs["road"]
        drivable_logits = road_outputs.get("drivable_logits")
        boundary_logits = road_outputs.get("boundary_logits")
        if (
            not isinstance(drivable_logits, Tensor)
            or not isinstance(boundary_logits, Tensor)
            or drivable_logits.ndim != 4
            or boundary_logits.ndim != 4
            or drivable_logits.shape[0] != batch_size
            or boundary_logits.shape[0] != batch_size
            or drivable_logits.shape[-2:] != boundary_logits.shape[-2:]
        ):
            raise ValueError("dynamic road outputs must share [B,H,W] geometry")

        device = reference.device
        entity_rows: list[dict[str, Tensor]] = []
        road_rows: list[dict[str, Tensor]] = []
        for index, (entity_target, road_target, record, image_size) in enumerate(
            zip(dynamic.entity, dynamic.road, records, image_sizes)
        ):
            entity_rows.append(
                entity_attribute_targets_from_p1(
                    entity_target, record.detections, device=device
                )
            )
            road_row = road_grounding_tensor_targets(
                road_target,
                image_size=image_size,
                output_size=tuple(int(value) for value in drivable_logits.shape[-2:]),
                device=device,
            )
            style = build_boundary_style_targets(
                road_target.boundaries,
                device=device,
                image_width=float(image_size[0]),
            )
            style_valid = style["valid_mask"] & road_row["boundary_valid_mask"]
            style_targets = style["targets"].clone()
            style_targets[~style_valid] = -1
            road_rows.append(
                {
                    **road_row,
                    "boundary_style_targets": style_targets,
                    "boundary_style_valid_mask": style_valid,
                }
            )

        targets = {
            "entity": self._concatenate_target_mappings(entity_rows),
            "road": self._concatenate_target_mappings(road_rows),
        }
        self.last_dynamic_grounding_batch = dynamic
        self.last_dynamic_grounding_targets = _checkpoint_clone(targets)
        return targets

    def _resolve_grounding_targets(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        *,
        reference: Tensor,
    ) -> Mapping[str, Any]:
        mode = batch.get("grounding_mode", "dynamic")
        if mode == "dynamic":
            return self._build_dynamic_grounding_targets(
                outputs, batch, reference=reference
            )
        if mode == "synthetic_prebuilt":
            targets = batch.get("grounding_targets")
            if not isinstance(targets, Mapping):
                raise ValueError(
                    "synthetic_prebuilt grounding mode requires grounding_targets"
                )
            self.last_dynamic_grounding_batch = None
            self.last_dynamic_grounding_targets = None
            self.last_dynamic_reliability = None
            return targets
        raise ValueError(f"unsupported grounding_mode {mode!r}")

    def _build_dynamic_reliability(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> DynamicReliabilityResult:
        if self.last_dynamic_grounding_batch is None:
            raise RuntimeError("dynamic reliability requires current dynamic grounding")
        records = batch.get("grounding_records")
        file_names = batch.get("file_names")
        if (
            not isinstance(records, (tuple, list))
            or not isinstance(file_names, (tuple, list))
            or len(records) != len(file_names)
        ):
            raise ValueError("dynamic reliability requires records and file_names")
        mirror_pairs = batch.get("mirror_pairs")
        if mirror_pairs is None:
            mirror_pairs = torch.empty(0, 2, dtype=torch.long)
        result = build_dynamic_reliability(
            outputs,
            self.last_dynamic_grounding_batch,
            records,
            targets["road"],
            mirror_pairs=mirror_pairs,
            sample_ids=tuple(str(value) for value in file_names),
            ema_state=self._view_ema_state,
        )
        self._view_ema_state = _checkpoint_clone(result.ema_state)
        self.last_dynamic_reliability = result
        return result

    @staticmethod
    def _grounding_loss(
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any] | None,
        mirror_pairs: Tensor | None,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Compute F02 from current-graph predictions and static supervision.

        Precomputed scalar loss terms are deliberately unsupported: they hide
        disconnected grounding heads and make admission statistics meaningless.
        """

        if targets is None:
            raise ValueError("P17 batch requires grounding_targets")
        grounding_outputs = outputs.get("grounding_outputs")
        if not isinstance(grounding_outputs, Mapping):
            raise ValueError("P17 requires current-graph outputs['grounding_outputs']")
        if not isinstance(targets.get("entity"), Mapping) or not isinstance(
            targets.get("road"), Mapping
        ):
            raise ValueError("grounding_targets must contain entity and road mappings")

        device = reference.device

        def on_device(values: Mapping[str, Any]) -> dict[str, Any]:
            return {
                name: value.to(device=device) if isinstance(value, Tensor) else value
                for name, value in values.items()
            }

        entity_results = entity_attribute_grounding_loss(
            grounding_outputs["entity"], on_device(targets["entity"])
        )
        road_output = grounding_outputs["road"]
        road_target = on_device(targets["road"])
        road_results = road_grounding_loss_bundle(
            drivable_logits=road_output["drivable_logits"],
            drivable_targets=road_target["drivable_targets"],
            drivable_valid_mask=road_target["drivable_valid_mask"],
            boundary_logits=road_output["boundary_logits"],
            boundary_targets=road_target["boundary_targets"],
            boundary_valid_mask=road_target["boundary_valid_mask"],
            boundary_style_logits=road_output["boundary_style_logits"],
            boundary_style_targets=road_target["boundary_style_targets"],
            boundary_style_valid_mask=road_target[
                "boundary_style_valid_mask"
            ],
            drivable_reliability=road_output.get("drivable_reliability"),
            boundary_reliability=road_output.get("boundary_reliability"),
        )
        entity = sum(
            (result.loss for result in entity_results.values()), _zero(reference)
        )
        road = sum(
            (result.loss for result in road_results.values()), _zero(reference)
        )

        masks = outputs.get("slot_masks")
        if not isinstance(masks, Tensor) or masks.ndim != 4 or masks.shape[1] != 20:
            raise ValueError("grounding view requires slot_masks [B,20,H,W]")
        if mirror_pairs is None:
            mirror_pairs = torch.empty(0, 2, dtype=torch.long, device=device)
        if (
            not isinstance(mirror_pairs, Tensor)
            or mirror_pairs.ndim != 2
            or mirror_pairs.shape[1] != 2
        ):
            raise ValueError("mirror_pairs must be integer [M,2]")
        pairs = mirror_pairs.to(device=device, dtype=torch.long)
        if pairs.numel() and (
            bool((pairs < 0).any()) or bool((pairs >= masks.shape[0]).any())
        ):
            raise ValueError("mirror_pairs contain out-of-range batch indices")
        if pairs.numel():
            permutation = torch.tensor(
                [*range(12), 14, 13, 12, 16, 15, 17, 18, 19],
                device=device,
            )
            canonical = masks.index_select(0, pairs[:, 0])
            mirrored = (
                masks.index_select(0, pairs[:, 1])
                .index_select(1, permutation)
                .flip(-1)
            )
            road_view = (
                canonical[:, 12:17] - mirrored[:, 12:17]
            ).abs().mean()
            entity_view = (
                canonical[:, :12].sum(1) - mirrored[:, :12].sum(1)
            ).abs().mean()
            latent_view = (
                canonical[:, 17:20].sum(1) - mirrored[:, 17:20].sum(1)
            ).abs().mean()
            ground_view = (road_view + entity_view + latent_view) / 3.0
        else:
            ground_view = _zero(masks)

        latent = outputs.get("latent_slots")
        first_view = outputs.get("latent_feature_view_one")
        second_view = outputs.get("latent_feature_view_two")
        if (
            not isinstance(latent, Tensor)
            or latent.ndim != 3
            or latent.shape[1] != 3
            or not isinstance(first_view, Tensor)
            or not isinstance(second_view, Tensor)
            or first_view.shape != latent.shape
            or second_view.shape != latent.shape
        ):
            raise ValueError(
                "P17 requires latent slots and two distinct feature-dropout views"
            )
        normalised = F.normalize(latent.float(), dim=-1)
        similarity = torch.einsum("bjd,bkd->bjk", normalised, normalised)
        off_diagonal = ~torch.eye(
            3, dtype=torch.bool, device=similarity.device
        )
        slot_diversity = similarity[:, off_diagonal].square().mean()
        feature_view = (
            1.0
            - F.cosine_similarity(
                first_view.float(), second_view.float(), dim=-1
            )
        ).mean()
        grounding = entity + road + 0.10 * ground_view + 0.02 * slot_diversity
        return grounding, feature_view, {
            "grounding_entity": entity,
            "grounding_road": road,
            "grounding_view": ground_view,
            "grounding_slot_diversity": slot_diversity,
            "feature_view": feature_view,
        }

    def _pu_inputs(self, outputs: Mapping[str, Any], reason_targets: Tensor, *, epoch: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        diagnostics = outputs.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or not isinstance(diagnostics.get("pu"), Mapping):
            raise ValueError("P17 requires P16 diagnostics['pu']")
        pu = diagnostics["pu"]
        required = {"p_evidence", "p_private", "c_view", "c_obs"}
        missing = required.difference(pu)
        if missing:
            raise ValueError(f"P16 PU diagnostics missing {sorted(missing)}")
        active, values = self._set_epoch_pu_state(epoch)
        lambda_values = values.to(device=reason_targets.device)
        soft = build_pu_soft_targets(
            reason_targets,
            pu["p_evidence"],
            pu["p_private"],
            pu["c_view"],
            pu["c_obs"],
            lambda_values,
            update_index=self.optimizer_step,
        )
        private_delta = outputs.get("reason_private_delta")
        if not isinstance(private_delta, Tensor):
            raise ValueError("P17 requires P16 reason_private_delta")
        head = getattr(self.model, "pu_private_head", None)
        keep_one = getattr(self.model, "_pu_feature_keep_view_one", None)
        keep_two = getattr(self.model, "_pu_feature_keep_view_two", None)
        if not isinstance(head, nn.Module) or not isinstance(keep_one, Tensor) or not isinstance(keep_two, Tensor):
            raise ValueError("P17 requires P16 independent pu_private_head and feature-view buffers")
        # This replay is the only trainable P12 path.  The detached P11 delta
        # gives pu_private supervision without allowing PU loss to alter
        # action, semantic, or reason-private representation owners.
        logits_one = head(private_delta.detach(), keep_one)
        logits_two = head(private_delta.detach(), keep_two)
        private_logits = 0.5 * (logits_one + logits_two)

        confidence = self._reason_confidence(outputs, reason_targets)
        loss = reason_private_pu_loss(
            private_logits,
            soft["soft_targets"],
            confidence["positive_weight"],
            confidence["negative_weight"],
        )
        return loss, soft["soft_targets"], confidence["positive_weight"], confidence["negative_weight"]

    @staticmethod
    def _reason_confidence(outputs: Mapping[str, Any], reason_targets: Tensor) -> dict[str, Tensor]:
        required = {
            "reason_slot_weights",
            "slot_reliability",
            "reason_unary_contributions_raw",
            "slot_observability",
        }
        missing = required.difference(outputs)
        if missing:
            raise ValueError(f"P17 requires P16 reason confidence outputs {sorted(missing)}")
        diagnostics = outputs["diagnostics"]["pu"]
        return reason_confidence_weights(
            outputs["reason_slot_weights"][..., :20],
            outputs["slot_reliability"],
            outputs["reason_unary_contributions_raw"],
            diagnostics["p_private_view_one"],
            diagnostics["p_private_view_two"],
            outputs["slot_observability"].mean(dim=1, keepdim=True).expand_as(reason_targets),
        )

    def _isolated_pairwise_auxiliary(
        self,
        outputs: Mapping[str, Any],
        action_targets: Tensor,
        reason_soft_targets: Tensor,
        positive_weight: Tensor,
        negative_weight: Tensor,
    ) -> Tensor:
        """Replay pair heads with all representation inputs detached.

        The formal P16 model exposes the public tensors required by the P10
        heads.  Replaying only those heads gives pairwise owner gradients via
        the single final backward, while preventing the auxiliary from
        replacing or contaminating final-logit training paths.
        """

        if not all(hasattr(self.model, name) for name in ("action_pairwise", "reason_pairwise")):
            raise ValueError("P17 requires P10 action_pairwise and reason_pairwise owners")
        required = {
            "action_tokens", "semantic_reason_tokens", "reason_private_delta",
            "evidence_slots", "slot_masks",
            "slot_sector_probs", "action_slot_weights", "reason_slot_weights",
            "slot_reliability", "action_logits_global", "reason_logits_global",
        }
        missing = required.difference(outputs)
        if missing:
            raise ValueError(f"P17 pairwise owner replay missing {sorted(missing)}")
        action_replay = getattr(self.model.action_pairwise, "owner_isolated_auxiliary", None)
        reason_replay = getattr(self.model.reason_pairwise, "owner_isolated_auxiliary", None)
        if not callable(action_replay) or not callable(reason_replay):
            raise ValueError("P17 requires P10 owner_isolated_auxiliary on both pairwise owners")
        evidence_slots = outputs["evidence_slots"]
        if not isinstance(evidence_slots, Tensor) or evidence_slots.ndim != 3 or evidence_slots.shape[1] != 20:
            raise ValueError("P17 pairwise replay requires canonical evidence_slots [B,20,D]")
        sector_probs = outputs["slot_sector_probs"]
        if not isinstance(sector_probs, Mapping) or not isinstance(sector_probs.get("horizontal"), Tensor):
            raise ValueError("P17 pairwise replay requires horizontal slot_sector_probs")
        shared_inputs = {
            "evidence_tokens": evidence_slots.detach(),
            "slot_masks": outputs["slot_masks"].detach(),
            "sector_probs": sector_probs["horizontal"].detach(),
            "reliability": outputs["slot_reliability"].detach(),
        }
        action_pair = action_replay(
            global_logits=outputs["action_logits_global"],
            target_tokens=outputs["action_tokens"],
            unary_public_pi=outputs["action_slot_weights"][..., :20],
            **shared_inputs,
        )
        reason_pair = reason_replay(
            global_logits=outputs["reason_logits_global"],
            target_tokens=outputs["semantic_reason_tokens"] + outputs["reason_private_delta"].detach(),
            unary_public_pi=outputs["reason_slot_weights"][..., :20],
            **shared_inputs,
        )
        action_aux_logits = action_pair.get("owner_auxiliary_logits")
        reason_aux_logits = reason_pair.get("owner_auxiliary_logits")
        if not isinstance(action_aux_logits, Tensor) or not isinstance(reason_aux_logits, Tensor):
            raise ValueError("P10 owner-isolated pair replay must return owner_auxiliary_logits")
        action_aux = multilabel_asymmetric_loss(action_aux_logits, action_targets)
        reason_aux = evidence_conditional_loss(
            reason_aux_logits,
            reason_soft_targets,
            positive_weight,
            negative_weight,
        )
        return action_aux + reason_aux

    def compute_loss_bundle(
        self,
        outputs: Mapping[str, Any],
        *,
        action_targets: Tensor,
        reason_targets: Tensor,
        grounding_targets: Mapping[str, Any] | None = None,
        mirror_pairs: Tensor | None = None,
        counterfactual_loss: Tensor | None = None,
        optimizer_step: int | None = None,
        epoch: int = 1,
    ) -> RAELLossBundle:
        step = self.optimizer_step if optimizer_step is None else int(optimizer_step)
        weights = self.schedule.weights(step)
        action_final = outputs["action_logits_final"]
        action_global = outputs["action_logits_global"]
        reason_final = outputs["reason_logits_final"]
        reason_global = outputs["reason_logits_global"]
        action_targets = action_targets.to(device=action_final.device, dtype=action_final.dtype)
        reason_targets = reason_targets.to(device=reason_final.device, dtype=reason_final.dtype)
        if action_targets.shape != action_final.shape or reason_targets.shape != reason_final.shape:
            raise ValueError("target shapes must exactly match formal action/reason logits")

        pu_private, soft_targets, positive_weight, negative_weight = self._pu_inputs(outputs, reason_targets, epoch=epoch)
        action_final_loss = multilabel_asymmetric_loss(action_final, action_targets)
        action_global_loss = multilabel_asymmetric_loss(action_global, action_targets)
        action_consistency = 0.05 * two_way_consistency_loss(action_final, action_global)
        action_soft_f1 = 0.05 * soft_f1_loss(action_final, action_targets)
        action = (
            action_final_loss
            + 0.5 * action_global_loss
            + action_consistency
            + action_soft_f1
        )
        reason_final_loss = evidence_conditional_loss(reason_final, soft_targets, positive_weight, negative_weight)
        reason_global_loss = multilabel_asymmetric_loss(reason_global, reason_targets)
        reason_ranking = 0.05 * multilabel_ranking_loss(reason_final, reason_targets)
        reason_consistency = 0.05 * two_way_consistency_loss(reason_final, reason_global)
        reason = (
            reason_final_loss
            + 0.5 * reason_global_loss
            + reason_ranking
            + reason_consistency
        )
        grounding, feature_view, grounding_components = self._grounding_loss(
            outputs,
            grounding_targets,
            mirror_pairs,
            action,
        )
        pairwise_auxiliary = self._isolated_pairwise_auxiliary(
            outputs, action_targets, soft_targets, positive_weight, negative_weight
        )
        pairwise_weighted = weights.pairwise_auxiliary * pairwise_auxiliary
        counterfactual = _zero(action) if counterfactual_loss is None else _scalar(counterfactual_loss, reference=action)
        non_regression = torch.relu(action_final_loss - action_global_loss.detach() + 0.002)
        total = sum(
            (
                action_final_loss,
                0.5 * action_global_loss,
                action_consistency,
                action_soft_f1,
                reason_final_loss,
                0.5 * reason_global_loss,
                reason_ranking,
                reason_consistency,
                weights.grounding * grounding,
                pairwise_weighted,
                weights.counterfactual * counterfactual,
                weights.non_regression * non_regression,
                weights.feature_view * feature_view,
                pu_private,
            ),
            _zero(action),
        )
        components = {
            "action": action,
            "reason": reason,
            "grounding": grounding,
            "pairwise_auxiliary": pairwise_auxiliary,
            "counterfactual": counterfactual,
            "non_regression": non_regression,
            "feature_view": feature_view,
            "pu_private": pu_private,
            "action_final": action_final_loss,
            "action_global": action_global_loss,
            "reason_final": reason_final_loss,
            "reason_global": reason_global_loss,
            "grounding_weighted": weights.grounding * grounding,
            "pairwise_auxiliary_weighted": pairwise_weighted,
            "counterfactual_weighted": weights.counterfactual * counterfactual,
            "non_regression_weighted": weights.non_regression * non_regression,
            "feature_view_weighted": weights.feature_view * feature_view,
            "total": total,
            **grounding_components,
        }
        return RAELLossBundle(
            action=action,
            reason=reason,
            grounding=grounding,
            pairwise_auxiliary=pairwise_auxiliary,
            counterfactual=counterfactual,
            non_regression=non_regression,
            feature_view=feature_view,
            pu_private=pu_private,
            total=total,
            weights=weights,
            components=components,
            pu_soft_targets=soft_targets.detach(),
        )

    def _owner_parameters(self, owner: str) -> list[nn.Parameter]:
        all_parameters = dict(self.model.named_parameters())
        return [all_parameters[name] for name in self.optimizer_bundle.owner_parameter_names[owner]]

    def _owner_parameter_map(self) -> dict[str, nn.Parameter]:
        all_parameters = dict(self.model.named_parameters())
        return {
            name: all_parameters[name]
            for names in self.optimizer_bundle.owner_parameter_names.values()
            for name in names
        }

    def _trainer_config(self) -> dict[str, Any]:
        return {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "total_optimizer_updates": self.schedule.total_optimizer_updates,
            "precision": self.precision,
            "owner_learning_rates": dict(sorted(self.optimizer_bundle.owner_learning_rates.items())),
        }

    def _accumulated_owner_grads_state(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "is_none": parameter.grad is None,
                "tensor": None if parameter.grad is None else parameter.grad.detach().clone(memory_format=torch.preserve_format),
            }
            for name, parameter in self._owner_parameter_map().items()
        }

    @staticmethod
    def _handle_was_removed(handle: Any) -> bool:
        """Read PyTorch's removable-handle registry after the real backward."""

        ref = getattr(handle, "hooks_dict_ref", None)
        hook_dict = ref() if callable(ref) else None
        return hook_dict is None or handle.id not in hook_dict

    def _owner_norms(self) -> dict[str, float]:
        return {owner: _grad_norm(self._owner_parameters(owner)) for owner in self.optimizer_bundle.owner_parameter_names}

    def _snapshot_owner_parameters(self) -> dict[str, Tensor]:
        names = [name for entries in self.optimizer_bundle.owner_parameter_names.values() for name in entries]
        all_parameters = dict(self.model.named_parameters())
        return {name: all_parameters[name].detach().clone() for name in names}

    def _rezero_norms(self) -> dict[str, float]:
        all_parameters = dict(self.model.named_parameters())
        names_by_id = {id(parameter): name for name, parameter in all_parameters.items()}

        def explicit_parameter(module_name: str, attribute_name: str) -> str:
            module = getattr(self.model, module_name, None)
            parameter = getattr(module, attribute_name, None)
            if not isinstance(parameter, nn.Parameter):
                raise ValueError(f"P17 ReZero contract missing {module_name}.{attribute_name}")
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError(f"P17 ReZero parameter {module_name}.{attribute_name} is not registered")
            return name

        output_names = {
            "action_reason_bridge": {
                explicit_parameter("action_reason_bridge", "gamma_as_raw"),
            },
            "unary_contribution": {
                explicit_parameter("action_unary", "gamma_unary_raw"),
                explicit_parameter("reason_unary", "gamma_unary_raw"),
            },
            "pairwise_relation": {
                explicit_parameter("action_pairwise", "pair_output"),
                explicit_parameter("reason_pairwise", "pair_output"),
                explicit_parameter("action_pairwise", "gamma_pair_raw"),
                explicit_parameter("reason_pairwise", "gamma_pair_raw"),
            },
        }

        def split(owner: str) -> tuple[float, float]:
            owner_names = set(self.optimizer_bundle.owner_parameter_names[owner])
            outputs = output_names[owner]
            if not outputs.issubset(owner_names):
                raise ValueError(f"P17 ReZero output parameters are not owned by {owner}")
            output_parameters = [all_parameters[name] for name in sorted(outputs)]
            internal_parameters = [all_parameters[name] for name in sorted(owner_names - outputs)]
            if not internal_parameters:
                raise ValueError(f"P17 ReZero owner {owner} has no internal parameters")
            return _grad_norm(output_parameters), _grad_norm(internal_parameters)

        bridge_out, bridge_inner = split("action_reason_bridge")
        unary_out, unary_inner = split("unary_contribution")
        pair_out, pair_inner = split("pairwise_relation")
        return {
            "bridge_output": bridge_out,
            "unary_output": unary_out,
            "pairwise_output": pair_out,
            "bridge_internal": bridge_inner,
            "unary_internal": unary_inner,
            "pairwise_internal": pair_inner,
        }

    def _autocast_context(self, images: Tensor):
        if images.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _canonical_evidence_boundary(outputs: Mapping[str, Any]) -> Tensor:
        """Read P16's explicit upstream public-slot boundary and fail closed."""

        direct = outputs.get("evidence_slots")
        if isinstance(direct, Tensor) and direct.ndim == 3 and direct.shape[1] == 20:
            return direct
        raise ValueError("P17 requires outputs['evidence_slots'] as the canonical upstream [B,20,D] boundary")

    @staticmethod
    def _admission_summary(admission: Any) -> dict[str, dict[str, float]]:
        """Serialize the actual P13 boundary tensors without inventing zeros.

        P18 needs a compact epoch record.  The P13 object remains the source
        of truth: each finite tensor attribute becomes a named norm and every
        compatible pair receives a real cosine.  A changed P13 public object
        therefore fails closed instead of silently turning into an empty audit.
        """

        tensors: dict[str, Tensor] = {}

        def collect(value: Any, prefix: str) -> None:
            if isinstance(value, Tensor):
                if value.numel() > 0 and bool(torch.isfinite(value.detach()).all()):
                    tensors[prefix] = value.detach().float().reshape(-1)
                return
            if isinstance(value, Mapping):
                for name, child in value.items():
                    collect(child, f"{prefix}.{name}" if prefix else str(name))
                return
            if hasattr(value, "__dict__"):
                for name, child in vars(value).items():
                    collect(child, f"{prefix}.{name}" if prefix else str(name))

        collect(admission, "")
        if not tensors:
            raise RuntimeError("P13 admission exposed no finite public gradient tensors")
        raw_norms = {name: float(value.norm().item()) for name, value in tensors.items()}
        names = tuple(sorted(tensors))
        cosines: dict[str, float] = {}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                first, second = tensors[left], tensors[right]
                if first.numel() != second.numel():
                    continue
                denominator = first.norm() * second.norm()
                if float(denominator.item()) > 0.0:
                    cosines[f"{left}__{right}"] = float((first @ second / denominator).item())
        if not cosines:
            raise RuntimeError("P13 admission exposed no comparable gradient pair for cosine audit")
        projected = {
            name: value
            for name, value in raw_norms.items()
            if "project" in name.lower() or "admit" in name.lower()
        }
        if not projected:
            projected = dict(raw_norms)
        return {
            "raw_norms": raw_norms,
            "projected_norms": projected,
            "cosines": cosines,
            "caps": dict(raw_norms),
            "ema_norms": dict(raw_norms),
        }

    def _scheduled_counterfactual(
        self,
        outputs: dict[str, Any],
        next_optimizer_step: int,
        *,
        visual_field: Any,
        batch: Mapping[str, Any],
    ) -> Tensor | None:
        if (
            next_optimizer_step < 1
            or next_optimizer_step % COUNTERFACTUAL_EVERY_UPDATES != 0
        ):
            return None
        if self.counterfactual_loss_fn is not None:
            return self.counterfactual_loss_fn(outputs, next_optimizer_step)
        if visual_field is None:
            raise RuntimeError(
                "scheduled formal counterfactual requires one-DINO visual_field"
            )
        build_replay = getattr(self.model, "build_counterfactual_replay", None)
        if not callable(build_replay):
            raise RuntimeError(
                "formal model must implement build_counterfactual_replay"
            )
        case_ids = batch.get("file_names")
        if not isinstance(case_ids, Sequence) or len(case_ids) != outputs[
            "action_logits_final"
        ].shape[0]:
            raise ValueError(
                "scheduled counterfactual requires one file_names entry per sample"
            )
        event_index = next_optimizer_step // COUNTERFACTUAL_EVERY_UPDATES - 1
        target_family = "action" if event_index % 2 == 0 else "reason"
        base_key = f"{target_family}_logits_final"
        deletion_key = f"{target_family}_analytical_deletion"
        replay = build_replay(
            visual_field,
            outputs,
            target_family=target_family,
        )
        required = {"shared_field", "public_readout", "public_contribution"}
        missing = required.difference(replay)
        if missing:
            raise ValueError(
                f"counterfactual replay missing {sorted(missing)}"
            )
        result = run_feature_intervention(
            optimizer_update=next_optimizer_step,
            shared_field=replay["shared_field"],
            slot_masks=outputs["slot_masks"],
            sector_probs=outputs["slot_sector_probs"]["horizontal"],
            base_logits=outputs[base_key],
            analytical_deletion=outputs[deletion_key],
            public_readout=replay["public_readout"],
            public_contribution=replay["public_contribution"],
            case_ids=case_ids,
        )
        self.last_counterfactual_result = result
        loss = result.get("loss")
        if not result["available"]:
            return _zero(outputs[base_key])
        if not isinstance(loss, Tensor):
            raise RuntimeError("available counterfactual must provide tensor loss")
        return loss

    @staticmethod
    def _finite_scalar_from_tensor(value: Any, *, name: str, reduction: str = "mean") -> float:
        if not isinstance(value, Tensor) or value.numel() == 0 or not bool(torch.isfinite(value.detach()).all()):
            raise ValueError(f"P18 mechanism observation requires finite tensor {name}")
        tensor = value.detach().float()
        if reduction == "rms":
            return float(tensor.square().mean().sqrt().item())
        return float(tensor.mean().item())

    def _mechanism_observation(
        self,
        *,
        outputs: Mapping[str, Any],
        bundle: RAELLossBundle,
        data_time: float,
        forward_time: float,
        backward_time: float,
        optimizer_time: float,
    ) -> dict[str, Any]:
        """Capture P18 quantities from this exact real train forward/update."""

        required = {
            "slot_masks", "slot_area", "slot_type_probs", "slot_reliability",
            "action_logits_global", "reason_logits_global", "action_unary_contributions",
            "reason_unary_contributions", "action_pairwise_contributions",
            "reason_pairwise_contributions", "semantic_reason_tokens", "reason_private_delta",
            "named_contribution_ratio", "latent_contribution_ratio", "background_mask",
            "layer_weights_action", "layer_weights_reason", "layer_weights_slots",
        }
        missing = required.difference(outputs)
        if missing:
            diagnostics = outputs.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise ValueError(f"P18 mechanism observation missing {sorted(missing)}")
            observed_count = diagnostics.get("dino_call_count")
            if isinstance(observed_count, Tensor) and observed_count.numel() == 1:
                observed_count = int(observed_count.detach().item())
            if isinstance(observed_count, bool) or not isinstance(observed_count, int):
                observed_count = None
            return {
                "p18_available": False,
                "p18_missing": sorted(missing),
                "data_time": float(data_time),
                "dino_call_count": observed_count,
                "field_time": float(forward_time),
                "backward_time": float(backward_time),
                "optimizer_time": float(optimizer_time),
            }
        masks = outputs["slot_masks"]
        if not isinstance(masks, Tensor) or masks.ndim != 4 or masks.shape[1] != 20:
            raise ValueError("P18 mechanism observation requires slot_masks [B,20,H,W]")
        flattened = masks.detach().float().flatten(start_dim=-2).clamp_min(0.0)
        normalized = flattened / flattened.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        slot_entropy = float((-(normalized * normalized.clamp_min(1.0e-8).log()).sum(dim=-1)).mean().item())
        pair_iou = torch.einsum("bsh,bth->bst", normalized, normalized)
        pair_iou = pair_iou[:, ~torch.eye(20, dtype=torch.bool, device=pair_iou.device)].mean()
        type_probs = outputs["slot_type_probs"].detach().float().clamp_min(1.0e-8)
        type_entropy = float((-(type_probs * type_probs.log()).sum(dim=-1)).mean().item())
        layer_values = torch.cat(
            (
                outputs["layer_weights_action"].detach().float().reshape(-1, 4),
                outputs["layer_weights_reason"].detach().float().reshape(-1, 4),
                outputs["layer_weights_slots"].detach().float().reshape(-1, 4),
            ),
            dim=0,
        ).clamp_min(1.0e-8)
        layer_entropy = float((-(layer_values * layer_values.log()).sum(dim=-1)).mean().item())
        diagnostics = outputs.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or not isinstance(diagnostics.get("collapse"), Mapping):
            raise ValueError("P18 mechanism observation requires formal collapse diagnostics")
        collapse = diagnostics["collapse"]
        gamma_modules = {
            "gamma_AS": ("action_reason_bridge", "gamma_as_raw"),
            "gamma_RA": ("reason_private", "gamma_ra_raw"),
            "gamma_unary": ("action_unary", "gamma_unary_raw"),
            "gamma_pairwise": ("action_pairwise", "gamma_pair_raw"),
        }
        gammas: dict[str, float] = {}
        for name, (module_name, parameter_name) in gamma_modules.items():
            parameter = getattr(getattr(self.model, module_name, None), parameter_name, None)
            gammas[name] = self._finite_scalar_from_tensor(parameter, name=f"{module_name}.{parameter_name}")
        positive_weight = self._finite_scalar_from_tensor(bundle.pu_soft_targets, name="pu_soft_targets")
        dynamic_reliability = self.last_dynamic_reliability
        q_view_source_counts = {"unavailable": 20 * int(masks.shape[0])}
        q_view_bootstrap_count = 0
        rho_nonzero_rate = self._finite_scalar_from_tensor(
            (outputs["slot_reliability"].detach() > 0.0).float(),
            name="slot_reliability_nonzero",
        )
        if isinstance(dynamic_reliability, DynamicReliabilityResult):
            q_view_source_counts = {
                "unavailable": int((dynamic_reliability.q_view_source == 0).sum().item()),
                "ema": int((dynamic_reliability.q_view_source == 1).sum().item()),
                "mirror": int((dynamic_reliability.q_view_source == 2).sum().item()),
                "feature_dropout": int(
                    (dynamic_reliability.q_view_source == 3).sum().item()
                ),
            }
            q_view_bootstrap_count = int(dynamic_reliability.q_view_bootstrap_count)
            rho_nonzero_rate = float(dynamic_reliability.rho_nonzero_rate)
        if not math.isfinite(float(data_time)) or float(data_time) < 0.0:
            raise ValueError("P18 mechanism observation requires real nonnegative data_time")
        return {
            "p18_available": True,
            "data_time": float(data_time),
            "dino_call_count": int(diagnostics["dino_call_count"]),
            "field_time": float(forward_time), "slot_time": 0.0, "category_time": 0.0,
            "relation_time": 0.0, "backward_time": float(backward_time), "optimizer_time": float(optimizer_time),
            "action_global_loss": self._finite_scalar_from_tensor(bundle.components["action_global"], name="action_global_loss"),
            "action_final_loss": self._finite_scalar_from_tensor(bundle.components["action_final"], name="action_final_loss"),
            "reason_global_loss": self._finite_scalar_from_tensor(bundle.components["reason_global"], name="reason_global_loss"),
            "reason_final_loss": self._finite_scalar_from_tensor(bundle.components["reason_final"], name="reason_final_loss"),
            "action_global_logit_rms": self._finite_scalar_from_tensor(outputs["action_logits_global"], name="action_logits_global", reduction="rms"),
            "reason_global_logit_rms": self._finite_scalar_from_tensor(outputs["reason_logits_global"], name="reason_logits_global", reduction="rms"),
            "action_unary_rms_over_global": self._finite_scalar_from_tensor(outputs["action_unary_contributions"], name="action_unary", reduction="rms") / max(1.0e-8, self._finite_scalar_from_tensor(outputs["action_logits_global"], name="action_global", reduction="rms")),
            "action_pairwise_rms_over_global": self._finite_scalar_from_tensor(outputs["action_pairwise_contributions"], name="action_pairwise", reduction="rms") / max(1.0e-8, self._finite_scalar_from_tensor(outputs["action_logits_global"], name="action_global", reduction="rms")),
            "reason_unary_rms_over_global": self._finite_scalar_from_tensor(outputs["reason_unary_contributions"], name="reason_unary", reduction="rms") / max(1.0e-8, self._finite_scalar_from_tensor(outputs["reason_logits_global"], name="reason_global", reduction="rms")),
            "reason_pairwise_rms_over_global": self._finite_scalar_from_tensor(outputs["reason_pairwise_contributions"], name="reason_pairwise", reduction="rms") / max(1.0e-8, self._finite_scalar_from_tensor(outputs["reason_logits_global"], name="reason_global", reduction="rms")),
            **gammas,
            "active_entity_count": float((outputs["slot_reliability"][:, :12].detach() > 0.5).float().sum(dim=1).mean().item()),
            "background_mass": self._finite_scalar_from_tensor(outputs["background_mask"], name="background_mask"),
            "latent_mass": self._finite_scalar_from_tensor(masks[:, 17:20], name="latent_slot_masks"),
            "slot_mask_entropy": slot_entropy, "slot_pair_iou": float(pair_iou.item()),
            "slot_area_mean": self._finite_scalar_from_tensor(outputs["slot_area"], name="slot_area"),
            "slot_area_std": float(outputs["slot_area"].detach().float().std(unbiased=False).item()),
            "entity_type_entropy": type_entropy, "traffic_state_entropy": type_entropy,
            "road_coverage": self._finite_scalar_from_tensor(masks[:, 12:17], name="road_slot_masks"),
            "named_contribution_ratio": self._finite_scalar_from_tensor(outputs["named_contribution_ratio"]["action"], name="named_ratio"),
            "latent_contribution_ratio": self._finite_scalar_from_tensor(outputs["latent_contribution_ratio"]["action"], name="latent_ratio"),
            "global_contribution_ratio": float(1.0 - self._finite_scalar_from_tensor(outputs["named_contribution_ratio"]["action"], name="named_ratio") - self._finite_scalar_from_tensor(outputs["latent_contribution_ratio"]["action"], name="latent_ratio")),
            "layer_entropy": layer_entropy, "positive_weight_mean": positive_weight,
            "negative_weight_mean": 1.0 - positive_weight,
            "q_view_source_counts": q_view_source_counts,
            "q_view_bootstrap_count": q_view_bootstrap_count,
            "rho_nonzero_rate": rho_nonzero_rate,
            "slot_feature_dropout_consistency_mean": self._finite_scalar_from_tensor(
                outputs["slot_feature_dropout_consistency"],
                name="slot_feature_dropout_consistency",
            ),
            "pu_active_label_count": float(self.pu_active_labels.sum().item()),
            "pu_soft_positive_count": float(bundle.pu_soft_targets.detach().float().sum().item()),
            "semantic_private_norm_ratio": self._finite_scalar_from_tensor(outputs["reason_private_delta"], name="reason_private_delta", reduction="rms") / max(1.0e-8, self._finite_scalar_from_tensor(outputs["semantic_reason_tokens"], name="semantic_reason_tokens", reduction="rms")),
            "action_reason_context_norm": self._finite_scalar_from_tensor(outputs["semantic_reason_tokens"], name="semantic_reason_tokens", reduction="rms"),
            "action_layer_weights": {str(index): float(value) for index, value in enumerate(outputs["layer_weights_action"].detach().float().mean(dim=(0, 1)).tolist())},
            "reason_layer_weights": {str(index): float(value) for index, value in enumerate(outputs["layer_weights_reason"].detach().float().mean(dim=(0, 1)).tolist())},
            "slot_layer_weights": {str(index): float(value) for index, value in enumerate(outputs["layer_weights_slots"].detach().float().mean(dim=(0, 1)).tolist())},
            "layer_collapse": {str(index): float(value) for index, value in enumerate(collapse["layer_collapse_fail"].detach().float().flatten().tolist())},
        }

    def mechanism_stats_from_step(
        self,
        step: RAELStepResult,
        *,
        artifact_context: Mapping[str, Any],
        counterfactual: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Convert one public P17 result into a P18 mechanism row from live state."""

        observation = step.mechanism_observation
        admission = artifact_context.get("admission")
        if (
            not isinstance(observation, Mapping)
            or observation.get("p18_available") is not True
            or not isinstance(admission, Mapping)
        ):
            raise ValueError("P18 mechanism row requires real step observation and admission")
        required = {
            "raw_norms", "projected_norms", "cosines", "caps", "ema_norms"
        }
        if required.difference(admission):
            raise ValueError("P18 mechanism row is missing real admission maps")
        cf_required = {
            "analytical_selected_effect", "selected_effect", "control_effect",
            "wrong_effect", "sign_consistency", "valid_action_target_count",
            "valid_reason_target_count",
        }
        if cf_required.difference(counterfactual):
            raise ValueError("P18 mechanism row is missing real formal counterfactual fields")
        common = {
            "schema_version": artifact_context["schema_version"],
            "producer": "fate_oia.engine.train_acpr_rael_oia:mechanism_stats",
            "source_fingerprint_sha256": artifact_context["source_fingerprint_sha256"],
            "config_sha256": artifact_context["config_sha256"],
            "epoch": int(artifact_context["epoch"]),
            "microbatch_step": int(step.microbatch_step),
            "optimizer_step": int(step.optimizer_step),
            "data_time": float(observation["data_time"]),
            "dino_time": float(observation["field_time"]),
            "samples_per_sec": 0.0,
            "allocated_gb": float(torch.cuda.memory_allocated() / (1024 ** 3)) if torch.cuda.is_available() else 0.0,
            "reserved_gb": float(torch.cuda.memory_reserved() / (1024 ** 3)) if torch.cuda.is_available() else 0.0,
            "dino_call_count": 1,
            "optimizer_stepped": bool(step.optimizer_stepped),
            "owner_gradient_norms": {name: float(value) for name, value in step.owner_gradient_norms_pre_clip.items()},
            "owner_parameter_delta": {name: float(value) for name, value in step.owner_parameter_delta.items()},
            "slot_cos_action_reason": dict(admission["cosines"]),
            "slot_cos_action_grounding": dict(admission["cosines"]),
            "slot_cos_action_cf": dict(admission["cosines"]),
            "negative_rates": {"mean": float(observation["negative_weight_mean"])},
            "projection_rates": dict(admission["projected_norms"]),
            "admission_rates": dict(admission["raw_norms"]),
            "raw_norms": dict(admission["raw_norms"]),
            "projected_norms": dict(admission["projected_norms"]),
            "budget_hit_rates": dict(admission["caps"]),
            "ema_norms": dict(admission["ema_norms"]),
            "pu_lambda_by_label": {str(index): float(self.pu_lambda[index].item()) for index in range(REASON_COUNT)},
            "counterfactual_available": bool(counterfactual.get("available", True)),
            "counterfactual_reason": str(counterfactual.get("reason", "formal_128_case_audit")),
            "analytic_selected_effect": (
                float(counterfactual["analytical_selected_effect"])
                if counterfactual["analytical_selected_effect"] is not None
                else None
            ),
            "feature_selected_effect": (
                float(counterfactual["selected_effect"])
                if counterfactual["selected_effect"] is not None
                else None
            ),
            "control_effect": (
                float(counterfactual["control_effect"])
                if counterfactual["control_effect"] is not None
                else None
            ),
            "wrong_target_effect": (
                float(counterfactual["wrong_effect"])
                if counterfactual["wrong_effect"] is not None
                else None
            ),
            "sign_consistency": (
                float(counterfactual["sign_consistency"])
                if counterfactual["sign_consistency"] is not None
                else None
            ),
            "valid_action_target_count": float(counterfactual["valid_action_target_count"]),
            "valid_reason_target_count": float(counterfactual["valid_reason_target_count"]),
        }
        common.update({name: value for name, value in observation.items() if name != "data_time"})
        common["field_time"] = float(observation["field_time"])
        common["backward_time"] = float(observation["backward_time"])
        common["optimizer_time"] = float(observation["optimizer_time"])
        common["slot_time"] = float(observation["slot_time"])
        common["category_time"] = float(observation["category_time"])
        common["relation_time"] = float(observation["relation_time"])
        common["samples_per_sec"] = float(1.0 / max(1.0e-9, common["field_time"] + common["backward_time"] + common["optimizer_time"]))
        return common

    def train_microbatch(self, batch: Mapping[str, Any], *, epoch: int) -> RAELStepResult:
        required = {
            "images",
            "action_targets",
            "reason_targets",
        }
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"P17 batch missing {sorted(missing)}")
        images = batch["images"]
        if not isinstance(images, Tensor):
            raise TypeError("batch images must be a Tensor")
        started = time.perf_counter()
        prior_epoch = self.epoch
        optimizer_boundary = (self.microbatch_step + 1) % self.gradient_accumulation_steps == 0
        accumulated_grads_before = self._accumulated_owner_grads_state() if optimizer_boundary else None
        admission_before = _checkpoint_clone(self.admission.state_dict()) if optimizer_boundary else None
        model_before = _checkpoint_clone(self.model.state_dict()) if optimizer_boundary else None
        view_ema_before = _checkpoint_clone(self._view_ema_state) if optimizer_boundary else None
        forward_state_before = (
            _capture_forward_scalar_state(self.model) if optimizer_boundary else None
        )
        rng_before = _capture_rng_state() if optimizer_boundary else None
        self.epoch = int(epoch)
        with self._autocast_context(images):
            next_completed_update = self.optimizer_step + 1
            formal_cf_boundary = (
                optimizer_boundary
                and self.counterfactual_loss_fn is None
                and next_completed_update % COUNTERFACTUAL_EVERY_UPDATES == 0
            )
            visual_field = None
            if batch.get("grounding_mode", "dynamic") == "dynamic":
                encode = getattr(self.model, "encode_images", None)
                provisional_decode = getattr(
                    self.model, "decode_from_field_provisional", None
                )
                reliability_decode = getattr(
                    self.model, "decode_from_field_with_reliability", None
                )
                if not all(
                    callable(value)
                    for value in (encode, provisional_decode, reliability_decode)
                ):
                    raise RuntimeError(
                        "dynamic grounding requires encode/provisional/reliability decode"
                    )
                visual_field = encode(images)
                provisional_outputs = provisional_decode(visual_field)
                grounding_targets = self._resolve_grounding_targets(
                    provisional_outputs,
                    batch,
                    reference=provisional_outputs["action_logits_final"],
                )
                reliability = self._build_dynamic_reliability(
                    provisional_outputs, batch, grounding_targets
                )
                outputs = reliability_decode(
                    visual_field,
                    q_ground=reliability.q_ground,
                    q_view=reliability.q_view,
                    q_view_sector=reliability.q_view_sector,
                )
            elif formal_cf_boundary:
                encode = getattr(self.model, "encode_images", None)
                decode = getattr(self.model, "decode_from_field", None)
                if not callable(encode) or not callable(decode):
                    raise RuntimeError(
                        "formal scheduled counterfactual requires encode_images/decode_from_field"
                    )
                visual_field = encode(images)
                outputs = decode(visual_field)
            else:
                outputs = self.model(images)
            if batch.get("grounding_mode", "dynamic") != "dynamic":
                grounding_targets = self._resolve_grounding_targets(
                    outputs, batch, reference=outputs["action_logits_final"]
                )
            evidence_slots = self._canonical_evidence_boundary(outputs)
            cf_loss = (
                self._scheduled_counterfactual(
                    outputs,
                    next_completed_update,
                    visual_field=visual_field,
                    batch=batch,
                )
                if optimizer_boundary
                else None
            )
            bundle = self.compute_loss_bundle(
                outputs,
                action_targets=batch["action_targets"],
                reason_targets=batch["reason_targets"],
                grounding_targets=grounding_targets,
                mirror_pairs=batch.get("mirror_pairs"),
                counterfactual_loss=cf_loss,
                epoch=epoch,
            )
        forward_finished = time.perf_counter()
        action_for_admission = bundle.action
        reason_for_admission = bundle.reason
        grounding_for_admission = (
            bundle.weights.grounding * bundle.grounding
            + bundle.weights.feature_view * bundle.feature_view
        )
        cf_for_admission = bundle.weights.counterfactual * bundle.counterfactual
        admission = self.admission.admit_from_losses(
            evidence_slots=evidence_slots,
            semantic_reason_tokens=outputs["semantic_reason_tokens"],
            action_loss=action_for_admission,
            reason_loss=reason_for_admission,
            grounding_loss=grounding_for_admission if grounding_for_admission.requires_grad else None,
            counterfactual_loss=cf_for_admission if cf_for_admission.requires_grad else None,
        )
        self.last_admission_summary = self._admission_summary(admission)
        scaled = bundle.total / float(self.gradient_accumulation_steps)
        admission_registered_count = 0
        admission_triggered_count = 0
        admission_removed_count = 0
        with self.admission.replace_shared_boundary_gradients(
            evidence_slots=evidence_slots,
            semantic_reason_tokens=outputs["semantic_reason_tokens"],
            admission=admission,
            backward_scale=1.0 / float(self.gradient_accumulation_steps),
        ) as hooks:
            # P13 owns hook creation/removal.  P17 records the actual handles
            # registered for this backward and whether each fired before exit.
            registered_handles = tuple(hooks._handles)
            admission_registered_count = len(registered_handles)
            scaled.backward()
            admission_triggered_count = sum(self._handle_was_removed(handle) for handle in registered_handles)
        # ``__exit__`` clears any untriggered handles too, including exception
        # paths; a nonempty private handle list here is a lifecycle violation.
        admission_removed_count = admission_registered_count if not hooks._handles else 0
        backward_finished = time.perf_counter()
        pre = self._owner_norms()
        post = dict(pre)
        task_delta = {owner: 0.0 for owner in self.optimizer_bundle.owner_parameter_names}
        optimizer_effect_delta = {owner: 0.0 for owner in self.optimizer_bundle.owner_parameter_names}
        decay_only_delta = {owner: 0.0 for owner in self.optimizer_bundle.owner_parameter_names}
        if optimizer_boundary:
            rezero = self._rezero_norms()
            bootstrap_candidate = ReZeroBootstrapTracker()
            bootstrap_candidate.load_state_dict(self.bootstrap.state_dict())
            bootstrap_candidate.observe(self.optimizer_step, rezero)
            try:
                bootstrap_candidate.assert_satisfied()
            except Exception:
                if model_before is not None:
                    self.model.load_state_dict(model_before)
                if forward_state_before is not None:
                    _restore_forward_scalar_state(self.model, forward_state_before)
                if accumulated_grads_before is not None:
                    self._restore_accumulated_owner_grads(accumulated_grads_before)
                if admission_before is not None:
                    self.admission.load_state_dict(admission_before)
                if view_ema_before is not None:
                    self._view_ema_state = _checkpoint_clone(view_ema_before)
                self.epoch = prior_epoch
                if rng_before is not None:
                    _restore_rng_state(rng_before)
                raise
            before = self._snapshot_owner_parameters()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in self.optimizer.param_groups for parameter in group["params"] if parameter.grad is not None],
                GRAD_CLIP_NORM,
            )
            post = self._owner_norms()
            self.optimizer.step()
            self.scheduler.step()
            self.bootstrap.load_state_dict(bootstrap_candidate.state_dict())
            self.optimizer_step += 1
            after = dict(self.model.named_parameters())
            for owner, names in self.optimizer_bundle.owner_parameter_names.items():
                optimizer_effect_delta[owner] = _parameter_delta(before, after, names)
                if pre[owner] > 0.0:
                    task_delta[owner] = optimizer_effect_delta[owner]
                    self.owner_optimizer_step_count[owner] += 1
                else:
                    decay_only_delta[owner] = optimizer_effect_delta[owner]
            self.optimizer.zero_grad(set_to_none=True)
        optimizer_finished = time.perf_counter()
        self.microbatch_step += 1
        components = {name: value.detach() for name, value in bundle.components.items()}
        self._last_pu_soft_targets = bundle.pu_soft_targets.detach().float().cpu()
        mechanism_observation = self._mechanism_observation(
            outputs=outputs,
            bundle=bundle,
            data_time=float(batch.get("_data_time", 0.0)),
            forward_time=forward_finished - started,
            backward_time=backward_finished - forward_finished,
            optimizer_time=optimizer_finished - backward_finished,
        )
        return RAELStepResult(
            components=components,
            optimizer_stepped=optimizer_boundary,
            optimizer_step=self.optimizer_step,
            microbatch_step=self.microbatch_step,
            admission_hook_count=admission_triggered_count,
            owner_gradient_norms_pre_clip=pre,
            owner_gradient_norms_post_clip=post,
            owner_task_gradient_norms_pre_clip=dict(pre),
            owner_parameter_delta=task_delta,
            owner_optimizer_effect_delta=optimizer_effect_delta,
            owner_decay_only_parameter_delta=decay_only_delta,
            owner_optimizer_step_count=dict(self.owner_optimizer_step_count),
            admission_registered_count=admission_registered_count,
            admission_triggered_count=admission_triggered_count,
            admission_removed_count=admission_removed_count,
            mechanism_observation=mechanism_observation,
        )

    def prepare_counterfactual_handoff(
        self,
        batch: Mapping[str, Any],
    ) -> RAELEncodedFieldHandoff:
        """Encode once and expose a detached field/output pair for replay only."""

        images = batch.get("images")
        names = batch.get("file_names")
        if not isinstance(images, Tensor):
            raise TypeError("counterfactual handoff requires Tensor images")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise TypeError("counterfactual handoff requires one file name per image")
        if len(names) != int(images.shape[0]):
            raise ValueError("counterfactual handoff file_names length mismatch")
        encode = getattr(self.model, "encode_images", None)
        if not callable(encode):
            raise RuntimeError("formal model must expose encode_images for the public replay handoff")
        counter = getattr(getattr(self.model, "dino_extractor", None), "dino_call_count", None)
        before = int(counter) if isinstance(counter, int) else None
        prior_training = self.model.training
        prior_ema = _checkpoint_clone(self._view_ema_state)
        try:
            self.model.eval()
            with torch.no_grad(), self._autocast_context(images):
                visual_field = encode(images)
                if batch.get("grounding_mode", "dynamic") == "dynamic":
                    provisional_decode = getattr(self.model, "decode_from_field_provisional", None)
                    reliability_decode = getattr(self.model, "decode_from_field_with_reliability", None)
                    if not callable(provisional_decode) or not callable(reliability_decode):
                        raise RuntimeError("dynamic replay handoff requires provisional and reliability decoders")
                    provisional = provisional_decode(visual_field)
                    grounding_targets = self._resolve_grounding_targets(
                        provisional,
                        batch,
                        reference=provisional["action_logits_final"],
                    )
                    reliability = self._build_dynamic_reliability(
                        provisional, batch, grounding_targets
                    )
                    outputs = reliability_decode(
                        visual_field,
                        q_ground=reliability.q_ground,
                        q_view=reliability.q_view,
                        q_view_sector=reliability.q_view_sector,
                    )
                else:
                    decode = getattr(self.model, "decode_from_field", None)
                    if not callable(decode):
                        raise RuntimeError("formal model must expose decode_from_field for the public replay handoff")
                    outputs = decode(visual_field)
        finally:
            self._view_ema_state = prior_ema
            self.model.train(prior_training)
        after_counter = getattr(getattr(self.model, "dino_extractor", None), "dino_call_count", None)
        after = int(after_counter) if isinstance(after_counter, int) else None
        if before is not None and after is not None and after != before + 1:
            raise RuntimeError("public counterfactual handoff must perform exactly one DINO encode")
        return RAELEncodedFieldHandoff(
            visual_field=visual_field,
            outputs=outputs,
            file_names=tuple(str(name) for name in names),
            dino_call_count_before=before,
            dino_call_count_after=after,
        )

    def replay_counterfactual_from_encoded_field(
        self,
        handoff: RAELEncodedFieldHandoff,
        *,
        target_family: str,
        optimizer_update: int,
        every_optimizer_updates: int = COUNTERFACTUAL_EVERY_UPDATES,
    ) -> dict[str, Any]:
        """Run the real feature intervention without a second DINO encode."""

        if not isinstance(handoff, RAELEncodedFieldHandoff):
            raise TypeError("counterfactual replay requires RAELEncodedFieldHandoff")
        if target_family not in {"action", "reason"}:
            raise ValueError("target_family must be action or reason")
        if int(every_optimizer_updates) <= 0:
            raise ValueError("every_optimizer_updates must be positive")
        outputs = handoff.outputs
        build_replay = getattr(self.model, "build_counterfactual_replay", None)
        if not callable(build_replay):
            raise RuntimeError("formal model must implement build_counterfactual_replay")
        counter = getattr(getattr(self.model, "dino_extractor", None), "dino_call_count", None)
        before = int(counter) if isinstance(counter, int) else None
        replay = build_replay(handoff.visual_field, outputs, target_family=target_family)
        required = {"shared_field", "public_readout", "public_contribution"}
        missing = required.difference(replay)
        if missing:
            raise ValueError(f"counterfactual replay missing {sorted(missing)}")
        base_key = f"{target_family}_logits_final"
        deletion_key = f"{target_family}_analytical_deletion"
        result = run_feature_intervention(
            optimizer_update=int(optimizer_update),
            shared_field=replay["shared_field"],
            slot_masks=outputs["slot_masks"],
            sector_probs=outputs["slot_sector_probs"]["horizontal"],
            base_logits=outputs[base_key],
            analytical_deletion=outputs[deletion_key],
            public_readout=replay["public_readout"],
            public_contribution=replay["public_contribution"],
            case_ids=handoff.file_names,
            every_optimizer_updates=int(every_optimizer_updates),
        )
        after_counter = getattr(getattr(self.model, "dino_extractor", None), "dino_call_count", None)
        after = int(after_counter) if isinstance(after_counter, int) else None
        if before is not None and after is not None and after != before:
            raise RuntimeError("counterfactual replay made an unexpected additional DINO encode")
        return {**result, "replay_dino_call_count": 0}

    def run_epoch_counterfactual_audit(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        epoch: int,
        device: torch.device,
        required_cases: int = 128,
    ) -> dict[str, Any]:
        """Run the formal 128-case audit from real test batches and one field per batch.

        This is deliberately read-only: no optimizer update, no re-encoding for
        a replay, and no repetition of a training-only scheduled case.  The
        case list is deterministic because the test loader is ordered.
        """

        if int(required_cases) != 128:
            raise ValueError("formal counterfactual artifact requires exactly 128 cases")
        sample_ids: list[str] = []
        selected: list[float] = []
        control: list[float] = []
        wrong: list[float] = []
        analytical: list[float] = []
        selected_beats_control = 0
        valid_action = 0
        valid_reason = 0
        was_training = self.model.training
        self.model.eval()
        try:
            for batch in batches:
                if len(sample_ids) >= required_cases:
                    break
                names = batch.get("file_names")
                if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
                    raise TypeError("counterfactual audit requires real file_names")
                device_batch = {
                    name: value.to(device, non_blocking=True)
                    if name in {"images", "action_targets", "reason_targets"}
                    else value
                    for name, value in batch.items()
                }
                handoff = self.prepare_counterfactual_handoff(device_batch)
                remaining = required_cases - len(sample_ids)
                take = min(len(handoff.file_names), remaining)
                for local_index in range(take):
                    # every=1 makes the public intervention select this exact
                    # real row from the already encoded field.
                    result = self.replay_counterfactual_from_encoded_field(
                        handoff,
                        target_family="action" if (len(sample_ids) % 2 == 0) else "reason",
                        optimizer_update=local_index + 1,
                        every_optimizer_updates=1,
                    )
                    case_id = canonicalize_sample_id(handoff.file_names[local_index])
                    if case_id in sample_ids:
                        raise RuntimeError("formal counterfactual audit case ids must be unique")
                    sample_ids.append(case_id)
                    if result.get("available") is not True:
                        continue
                    if result.get("case_id") != case_id:
                        raise RuntimeError(
                            "counterfactual replay selected a different real case: "
                            f"expected={case_id!r}, actual={result.get('case_id')!r}, "
                            f"local_index={local_index}"
                        )
                    effects = result.get("effects")
                    if not isinstance(effects, Mapping):
                        raise RuntimeError("available counterfactual is missing real effects")
                    try:
                        selected_effect = float(effects["d_selected"].detach().float().item())
                        control_effect = float(effects["d_control"].detach().float().item())
                        selected.append(selected_effect)
                        control.append(control_effect)
                        wrong.append(float(effects["d_wrong"].detach().float().item()))
                        diagnostics = result.get("diagnostics")
                        if not isinstance(diagnostics, Mapping) or not isinstance(diagnostics.get("positive_analytical_effect"), Tensor):
                            raise RuntimeError("counterfactual replay lacks real analytical effect diagnostic")
                        analytical.append(float(diagnostics["positive_analytical_effect"].detach().float().item()))
                        selected_beats_control += int(selected_effect > control_effect)
                    except (KeyError, AttributeError) as error:
                        raise RuntimeError("counterfactual effect is not a real scalar tensor") from error
                    if len(sample_ids) % 2 == 1:
                        valid_action += 1
                    else:
                        valid_reason += 1
        finally:
            self.model.train(was_training)
        if len(sample_ids) != required_cases:
            raise RuntimeError("formal 128-case audit exhausted the real test loader")
        available = bool(selected and control and wrong)
        payload = {
            "available": available,
            "reason": "formal_128_case_audit" if available else "no_eligible_control",
            "sample_ids": tuple(sample_ids),
            "selected_effect": sum(selected) / len(selected) if available else None,
            "control_effect": sum(control) / len(control) if available else None,
            "wrong_effect": sum(wrong) / len(wrong) if available else None,
            "analytical_selected_effect": sum(analytical) / len(analytical) if available else None,
            "sign_consistency": selected_beats_control / len(selected) if available else None,
            "valid_action_target_count": valid_action,
            "valid_reason_target_count": valid_reason,
            "epoch": int(epoch),
        }
        self.last_counterfactual_result = payload
        return payload

    def train_epoch_and_publish(
        self,
        train_batches: Iterable[Mapping[str, Any]],
        *,
        epoch: int,
        test_batches: Iterable[Mapping[str, Any]],
        epoch_transition: Callable[[], Mapping[str, Any]],
        expected_test_split_hash: str,
        action_schema_path: str | Path,
        reason_schema_path: str | Path,
        device: torch.device,
        writer: RAELArtifactWriter,
        epoch_artifact_builder: Callable[..., Mapping[str, Any]],
        on_step_result: Callable[[RAELStepResult], None] | None = None,
        case_collector: object | None = None,
        case_export_provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train one epoch, test-evaluate it, then publish only P18-validated artifacts."""

        if not isinstance(writer, RAELArtifactWriter) or not callable(epoch_artifact_builder):
            raise TypeError("epoch publishing requires the real P18 writer and artifact builder")
        self.last_epoch_pu_soft_positive = torch.zeros(REASON_COUNT, dtype=torch.float32)
        last_step_result: RAELStepResult | None = None
        step_count = 0
        for batch in train_batches:
            last_step_result = self.train_microbatch(batch, epoch=epoch)
            step_count += 1
            if on_step_result is not None:
                on_step_result(last_step_result)
            if self._last_pu_soft_targets is None:
                raise RuntimeError("P17 did not expose real PU soft targets for this training step")
            self.last_epoch_pu_soft_positive += self._last_pu_soft_targets.sum(dim=0)
        if last_step_result is None:
            raise ValueError("epoch publishing requires nonempty train batches")
        # The transition is intentionally after representation learning.  It
        # owns fixed train-audit PU gating followed by train-calib fitting;
        # neither operation may see test rows or precede this epoch's updates.
        transition = epoch_transition()
        if not isinstance(transition, Mapping):
            raise TypeError("epoch transition must return real train-audit/calibration state")
        action_calibration = transition.get("action_calibration")
        reason_calibration = transition.get("reason_calibration")
        expected_train_calib_split_hash = transition.get("train_calib_split_hash")
        pu_audit = transition.get("pu_audit")
        if not isinstance(action_calibration, Mapping) or not isinstance(reason_calibration, Mapping):
            raise ValueError("epoch transition omitted real train-calib calibrations")
        if not isinstance(expected_train_calib_split_hash, str) or not expected_train_calib_split_hash:
            raise ValueError("epoch transition omitted the train-calib split hash")
        if not isinstance(pu_audit, Sequence) or isinstance(pu_audit, (str, bytes)) or len(pu_audit) != REASON_COUNT:
            raise ValueError("epoch transition omitted the 21-row fixed train-audit PU result")
        if iter(test_batches) is test_batches:
            raise TypeError("epoch publishing requires a re-iterable test loader, never a cached or one-shot generator")
        counterfactual = self.run_epoch_counterfactual_audit(test_batches, epoch=epoch, device=device)
        evaluation = evaluate_rael_test_only(
            model=self.model,
            batches=test_batches,
            action_calibration=action_calibration,
            reason_calibration=reason_calibration,
            expected_train_calib_split_hash=expected_train_calib_split_hash,
            expected_test_split_hash=expected_test_split_hash,
            action_schema_path=action_schema_path,
            reason_schema_path=reason_schema_path,
            device=device,
            case_collector=case_collector,
            case_export_provenance=case_export_provenance,
        )
        artifacts = epoch_artifact_builder(
            trainer=self,
            epoch=int(epoch),
            last_step_result=last_step_result,
            step_count=step_count,
            evaluation=evaluation,
            counterfactual=counterfactual,
            pu_audit=tuple(dict(row) for row in pu_audit),
            action_calibration=action_calibration,
            reason_calibration=reason_calibration,
        )
        records = writer.write_epoch(int(epoch), artifacts)
        return {
            "last_step_result": last_step_result,
            "step_count": step_count,
            "pu_audit": tuple(dict(row) for row in pu_audit),
            "evaluation": evaluation,
            "artifact_records": records,
        }

    def loss_owner_gradient_matrix(self, batch: Mapping[str, Any], *, epoch: int) -> dict[str, dict[str, float]]:
        """Read-only parameter-level firewall audit; never writes ``.grad``."""

        if batch.get("grounding_mode", "dynamic") == "dynamic":
            field = self.model.encode_images(batch["images"])
            provisional = self.model.decode_from_field_provisional(field)
            grounding_targets = self._resolve_grounding_targets(
                provisional, batch, reference=provisional["action_logits_final"]
            )
            reliability = self._build_dynamic_reliability(
                provisional, batch, grounding_targets
            )
            outputs = self.model.decode_from_field_with_reliability(
                field,
                q_ground=reliability.q_ground,
                q_view=reliability.q_view,
                q_view_sector=reliability.q_view_sector,
            )
        else:
            outputs = self.model(batch["images"])
            grounding_targets = self._resolve_grounding_targets(
                outputs, batch, reference=outputs["action_logits_final"]
            )
        bundle = self.compute_loss_bundle(
            outputs,
            action_targets=batch["action_targets"],
            reason_targets=batch["reason_targets"],
            grounding_targets=grounding_targets,
            mirror_pairs=batch.get("mirror_pairs"),
            epoch=epoch,
        )
        losses = {"action": bundle.action, "reason": bundle.reason, "pu_private": bundle.pu_private}
        result: dict[str, dict[str, float]] = {}
        all_parameters = dict(self.model.named_parameters())
        for loss_name, loss in losses.items():
            if not loss.requires_grad:
                result[loss_name] = {owner: 0.0 for owner in self.optimizer_bundle.owner_parameter_names}
                continue
            params = [all_parameters[name] for names in self.optimizer_bundle.owner_parameter_names.values() for name in names]
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            by_name = {name: grad for name, grad in zip((name for names in self.optimizer_bundle.owner_parameter_names.values() for name in names), grads)}
            result[loss_name] = {
                owner: _grad_norm_from_tensors([by_name[name] for name in names])
                for owner, names in self.optimizer_bundle.owner_parameter_names.items()
            }
        return result

    def state_dict(self) -> dict[str, Any]:
        rng = _capture_rng_state()
        return {
            "checkpoint_schema": "rael-p17-resume-v3",
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()),
            "admission": copy.deepcopy(self.admission.state_dict()),
            "owner_parameter_names": copy.deepcopy(self.optimizer_bundle.owner_parameter_names),
            "trainer_config": self._trainer_config(),
            "resume_fingerprints": dict(self._resume_fingerprints),
            "accumulated_owner_grads": self._accumulated_owner_grads_state(),
            "view_ema_state": _checkpoint_clone(self._view_ema_state),
            "posthoc_calibration_state": _checkpoint_clone(self._posthoc_calibration_state),
            "epoch": self.epoch,
            "microbatch_step": self.microbatch_step,
            "optimizer_step": self.optimizer_step,
            "pu_lambda": self.pu_lambda.clone(),
            "pu_active_labels": self.pu_active_labels.clone(),
            "owner_optimizer_step_count": copy.deepcopy(self.owner_optimizer_step_count),
            "bootstrap": self.bootstrap.state_dict(),
            **rng,
        }

    def _validate_resume_state(self, state: Mapping[str, Any]) -> None:
        required = {
            "checkpoint_schema",
            "owner_parameter_names",
            "trainer_config",
            "resume_fingerprints",
            "accumulated_owner_grads",
            "view_ema_state",
            "posthoc_calibration_state",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"P17 resume checkpoint missing required fields: {sorted(missing)}")
        if state["checkpoint_schema"] != "rael-p17-resume-v3":
            raise ValueError("P17 checkpoint schema mismatch")

        checkpoint_owners = {
            str(owner): tuple(str(name) for name in names)
            for owner, names in state["owner_parameter_names"].items()
        }
        if checkpoint_owners != self.optimizer_bundle.owner_parameter_names:
            raise ValueError("P17 resume owner parameter names do not match the current owner topology")

        checkpoint_config = state["trainer_config"]
        current_config = self._trainer_config()
        for name, current_value in current_config.items():
            if checkpoint_config.get(name) != current_value:
                raise ValueError(f"P17 resume {name} mismatch")

        try:
            checkpoint_fingerprints = _validate_rael_fingerprint_manifest(
                state["resume_fingerprints"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"P17 resume fingerprint manifest invalid: {error}") from error
        if checkpoint_fingerprints != self._resume_fingerprints:
            raise ValueError("P17 resume fingerprint manifest mismatch")

        expected_names = set(self._owner_parameter_map())
        checkpoint_grads = state["accumulated_owner_grads"]
        if set(checkpoint_grads) != expected_names:
            raise ValueError("P17 resume accumulated gradients do not match owner parameters")
        for name, payload in checkpoint_grads.items():
            if not isinstance(payload, Mapping) or set(payload) != {"is_none", "tensor"}:
                raise ValueError(f"P17 resume accumulated gradient payload is invalid for {name}")
            if not isinstance(payload["is_none"], bool):
                raise ValueError(f"P17 resume accumulated gradient marker is invalid for {name}")
            if payload["is_none"] and payload["tensor"] is not None:
                raise ValueError(f"P17 resume accumulated gradient must be None for {name}")
            if not payload["is_none"] and not isinstance(payload["tensor"], Tensor):
                raise ValueError(f"P17 resume accumulated gradient tensor is missing for {name}")

    def _restore_accumulated_owner_grads(self, stored: Mapping[str, Mapping[str, Any]]) -> None:
        owner_parameters = self._owner_parameter_map()
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in owner_parameters.items():
            payload = stored[name]
            if bool(payload["is_none"]):
                parameter.grad = None
                continue
            tensor = payload["tensor"]
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise ValueError(f"P17 resume accumulated gradient shape mismatch for {name}")
            if not torch.is_floating_point(tensor):
                raise ValueError(f"P17 resume accumulated gradient dtype is invalid for {name}")
            parameter.grad = tensor.detach().to(device=parameter.device, dtype=parameter.dtype).clone(
                memory_format=torch.preserve_format
            )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._validate_resume_state(state)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.admission.load_state_dict(state["admission"])
        self.epoch = int(state["epoch"])
        self.microbatch_step = int(state["microbatch_step"])
        self.optimizer_step = int(state["optimizer_step"])
        self.pu_lambda = state["pu_lambda"].detach().float().cpu().clone()
        self.pu_active_labels = state["pu_active_labels"].detach().bool().cpu().clone()
        self.owner_optimizer_step_count = {name: int(value) for name, value in state["owner_optimizer_step_count"].items()}
        self.bootstrap.load_state_dict(state["bootstrap"])
        _restore_rng_state(state)
        self._view_ema_state = _checkpoint_clone(state["view_ema_state"])
        self._posthoc_calibration_state = _checkpoint_clone(state["posthoc_calibration_state"])
        self._restore_accumulated_owner_grads(state["accumulated_owner_grads"])
        self._set_epoch_pu_state(self.epoch)


def _grad_norm_from_tensors(gradients: Sequence[Tensor | None]) -> float:
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0


__all__ = [
    "COUNTERFACTUAL_EVERY_UPDATES",
    "OwnerOptimizerBundle",
    "RAELLossBundle",
    "RAELStepResult",
    "RAELTrainer",
    "RAELWarmupSchedule",
    "RAELWarmupWeights",
    "ReZeroBootstrapTracker",
    "build_rael_repository_fingerprints",
    "build_rael_optimizer",
    "rael_repository_fingerprint_files",
]
