from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
import yaml

from fate_oia.models.rael_oia_model import BRANCH_NAMES as MODEL_BRANCH_NAMES


def _module():
    try:
        return importlib.import_module("fate_oia.utils.rael_artifacts")
    except ModuleNotFoundError as error:
        pytest.fail(f"P18 RED: artifact owner is absent: {error}")


RUN_ROOT_FILES = {
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
}
P17_FINGERPRINT_GROUPS = ("source", "test", "config", "schema", "skill", "script")
OWNER_NAMES = (
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
PRIVATE_OWNER_NAMES = frozenset({"reason_private", "pu_private"})
EPOCH_FILES = {
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
}



def _p17_stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _p17_manifest_entries_hash(
    *,
    namespace: str,
    phase: str,
    paths: list[str],
    file_status: dict[str, str],
    file_sha256: dict[str, str | None],
) -> str:
    return _p17_stable_hash(
        {
            "namespace": namespace,
            "schema": "rael-repository-fingerprint-v4",
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


def _p17_v4_fingerprint() -> dict[str, object]:
    """Build a compact but exact P17-v4 six-group fingerprint fixture."""

    phase = "development"
    groups = {
        "source": ["fate_oia/models/rael_oia_model.py"],
        "test": ["tests/test_rael_artifacts.py"],
        "config": ["configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml"],
        "schema": ["configs/rael_slot_schema.yaml"],
        "skill": [".codex/skills/rael-oia-v1-implementation-audit/SKILL.md"],
        "script": ["scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1"],
    }
    assert tuple(groups) == P17_FINGERPRINT_GROUPS
    all_paths = [path for group in P17_FINGERPRINT_GROUPS for path in groups[group]]
    file_status = {path: "present" for path in all_paths}
    file_sha256 = {
        path: hashlib.sha256(f"fixture:{path}".encode("utf-8")).hexdigest()
        for path in all_paths
    }
    group_hashes = {
        group: _p17_manifest_entries_hash(
            namespace=f"rael-{group}-v4",
            phase=phase,
            paths=groups[group],
            file_status=file_status,
            file_sha256=file_sha256,
        )
        for group in P17_FINGERPRINT_GROUPS
    }
    return {
        "fingerprint_schema": "rael-repository-fingerprint-v4",
        "phase": phase,
        "complete": True,
        "groups": groups,
        "file_status": file_status,
        "file_sha256": file_sha256,
        "missing_files": [],
        "group_hashes": group_hashes,
        "source_hash": _p17_stable_hash(
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
        "required_files_hash": _p17_manifest_entries_hash(
            namespace="rael-required-declared-and-import-closure-v4",
            phase=phase,
            paths=all_paths,
            file_status=file_status,
            file_sha256=file_sha256,
        ),
    }


P17_FINGERPRINT = _p17_v4_fingerprint()


def _p17_fingerprint_with_source_change() -> dict[str, object]:
    """Return a second valid P17-v4 manifest with one real source-file difference."""

    fingerprint = copy.deepcopy(P17_FINGERPRINT)
    groups = fingerprint["groups"]
    file_status = fingerprint["file_status"]
    file_sha256 = fingerprint["file_sha256"]
    source_path = groups["source"][0]
    file_sha256[source_path] = hashlib.sha256(b"independent-builder-source").hexdigest()
    phase = fingerprint["phase"]
    group_hashes = {
        group: _p17_manifest_entries_hash(
            namespace=f"rael-{group}-v4",
            phase=phase,
            paths=groups[group],
            file_status=file_status,
            file_sha256=file_sha256,
        )
        for group in P17_FINGERPRINT_GROUPS
    }
    fingerprint["group_hashes"] = group_hashes
    fingerprint["source_hash"] = _p17_stable_hash(
        {
            "namespace": "rael-source-test-skill-script-v4",
            "phase": phase,
            "groups": {
                group: group_hashes[group]
                for group in ("source", "test", "skill", "script")
            },
        }
    )
    fingerprint["config_hash"] = group_hashes["config"]
    fingerprint["schema_hash"] = group_hashes["schema"]
    paths = [path for group in P17_FINGERPRINT_GROUPS for path in groups[group]]
    fingerprint["required_files_hash"] = _p17_manifest_entries_hash(
        namespace="rael-required-declared-and-import-closure-v4",
        phase=phase,
        paths=paths,
        file_status=file_status,
        file_sha256=file_sha256,
    )
    return fingerprint


def _recompute_p17_fingerprint(fingerprint: dict[str, object]) -> None:
    groups = fingerprint["groups"]
    file_status = fingerprint["file_status"]
    file_sha256 = fingerprint["file_sha256"]
    phase = fingerprint["phase"]
    group_hashes = {
        group: _p17_manifest_entries_hash(
            namespace=f"rael-{group}-v4",
            phase=phase,
            paths=groups[group],
            file_status=file_status,
            file_sha256=file_sha256,
        )
        for group in P17_FINGERPRINT_GROUPS
    }
    fingerprint["group_hashes"] = group_hashes
    fingerprint["source_hash"] = _p17_stable_hash(
        {
            "namespace": "rael-source-test-skill-script-v4",
            "phase": phase,
            "groups": {
                group: group_hashes[group]
                for group in ("source", "test", "skill", "script")
            },
        }
    )
    fingerprint["config_hash"] = group_hashes["config"]
    fingerprint["schema_hash"] = group_hashes["schema"]
    paths = [path for group in P17_FINGERPRINT_GROUPS for path in groups[group]]
    fingerprint["required_files_hash"] = _p17_manifest_entries_hash(
        namespace="rael-required-declared-and-import-closure-v4",
        phase=phase,
        paths=paths,
        file_status=file_status,
        file_sha256=file_sha256,
    )


SOURCE_SHA = str(P17_FINGERPRINT["required_files_hash"])
RESOLVED_CONFIG = {
    "training": {"epochs": 14},
    "runtime": {"test_only": True},
}
CONFIG_SHA = hashlib.sha256(
    json.dumps(
        RESOLVED_CONFIG,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
BRANCH_NAMES = tuple(MODEL_BRANCH_NAMES)


def _per_label_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "id": index,
            "name": f"label-{index}",
            "F1": 0.5,
            "AP": 0.6,
            "AUC": 0.7,
            "support": 1,
            "threshold": 0.0,
        }
        for index in range(count)
    ]


def _provenance(*, producer: str, epoch: int | None = None, sample_count: int | None = None):
    value: dict[str, object] = {
        "schema_version": "rael-artifact-v1",
        "producer": producer,
        "source_fingerprint_sha256": SOURCE_SHA,
        "config_sha256": CONFIG_SHA,
    }
    if epoch is not None:
        value["epoch"] = epoch
    if sample_count is not None:
        value["sample_count"] = sample_count
    return value


def _run_payload(name: str) -> dict[str, object]:
    base = _provenance(producer=f"fate_oia.tests.{name}")
    if name == "run_manifest.json":
        return {
            **base,
            "git_head": "c" * 40,
            "remote_head": "c" * 40,
            "base_head": "d" * 40,
            "command": ["python", "-m", "fate_oia.engine.train_acpr_rael_oia"],
            "data_split": {"name": "test", "ids_sha256": "e" * 64, "sample_count": 512},
            "dino": {"source_sha256": "f" * 64, "weight_sha256": "1" * 64},
            "formal_flags": {
                "direct_image": True,
                "feature_cache_enabled": False,
                "token_compression": "none",
                "test_only": True,
            },
            "selected_runtime_profile": "batch4_accum8_workers4",
            "seed": 20260725,
            "test_selected": True,
            "publication_eligible": False,
        }
    if name == "config_resolved.yaml":
        return {
            **base,
            "resolved_config": RESOLVED_CONFIG,
            "resolved_config_sha256": CONFIG_SHA,
        }
    if name == "source_fingerprint.json":
        return {
            **base,
            **P17_FINGERPRINT,
        }
    if name == "runtime_profile.json":
        return {**base, "candidates": [{"name": "b4a8", "samples_per_sec": 1.25}]}
    if name == "selected_runtime_profile.json":
        return {
            **base,
            "selected": {
                "name": "b4a8",
                "batch_size": 4,
                "gradient_accumulation_steps": 8,
                "num_workers": 4,
                "amortized_samples_per_sec": 1.25,
            },
            "reason": "fastest stable full-mechanism candidate",
        }
    if name == "optimizer_owners.json":
        return {
            **base,
            "owners": {
                owner: {
                    "parameter_names": [f"{owner}.weight", f"{owner}.bias"],
                    "parameters": [
                        {
                            "name": f"{owner}.weight",
                            "lr": 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002,
                            "weight_decay": 0.05,
                        },
                        {
                            "name": f"{owner}.bias",
                            "lr": 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002,
                            "weight_decay": 0.0,
                        },
                    ],
                    "lr": 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002,
                    "weight_decay": 0.05,
                    "count": 2,
                }
                for owner in OWNER_NAMES
            },
        }
    raise AssertionError(name)


def _json_epoch_payload(name: str, *, epoch: int, sample_count: int) -> dict[str, object]:
    base = _provenance(
        producer=f"fate_oia.engine.eval_acpr_rael_oia:{name}",
        epoch=epoch,
        sample_count=sample_count,
    )
    if name in {"raw_metrics.json", "deploy_metrics.json"}:
        return {
            **base,
            "metrics": {
                "action": {"mF1": 0.6, "oF1": 0.7, "mAP": 0.72, "AUC": 0.75},
                "reason": {"mF1": 0.3, "oF1": 0.4, "mAP": 0.5, "AUC": 0.6},
                "joint": 0.45,
            },
        }
    if name == "branch_metrics.json":
        return {
            **base,
            "branches": [
                {
                    "name": branch_name,
                    "config": {"diagnostic_mode": branch_name},
                    "metrics": {
                        "action": {"mF1": 0.6, "oF1": 0.7, "mAP": 0.72, "AUC": 0.75},
                        "reason": {"mF1": 0.3, "oF1": 0.4, "mAP": 0.5, "AUC": 0.6},
                        "joint": 0.1 + index * 0.001,
                    },
                    "per_action": _per_label_rows(4),
                    "per_reason": _per_label_rows(21),
                }
                for index, branch_name in enumerate(BRANCH_NAMES)
            ],
        }
    if name in {"per_action.json", "per_reason.json"}:
        row_count = 4 if name == "per_action.json" else 21
        return {
            **base,
            "rows": _per_label_rows(row_count),
        }
    if name == "slot_stats.json":
        return {
            **base,
            "slot_count": 20,
            "mass": {"named": 0.5, "latent": 0.2, "background": 0.3},
            "area": {"mean": 0.1, "std": 0.02},
            "entropy": 0.8,
            "iou": 0.1,
            "attributes": {"entity_type_entropy": 0.7},
            "reliability": {"mean": 0.6},
        }
    if name == "layer_stats.json":
        return {
            **base,
            "action_layer_weights": [[0.25] * 4 for _ in range(4)],
            "reason_layer_weights": [[0.25] * 4 for _ in range(21)],
            "slot_layer_weights": [[0.25] * 4 for _ in range(20)],
            "entropy": 1.0,
            "collapse": False,
        }
    if name == "relation_stats.json":
        return {
            **base,
            "unary": {"rms": 0.1},
            "pairwise": {"rms": 0.05},
            "null": {"mass": 0.2},
            "alpha": {"mean": 0.1},
            "active_pair_count": 12,
            "total_pair_count": 190,
        }
    if name == "contribution_stats.json":
        return {
            **base,
            "global": {"rms": 1.0},
            "unary": {"rms": 0.1},
            "pairwise": {"rms": 0.05},
            "positive": {"mass": 0.08},
            "negative": {"mass": 0.04},
            "reconstruction_error": 1.0e-7,
        }
    if name == "named_latent_global.json":
        return {
            **base,
            "named_ratio": 0.5,
            "latent_ratio": 0.2,
            "global_ratio": 0.3,
            "per_target": [{"target": 0, "named": 0.5, "latent": 0.2, "global": 0.3}],
            "overall": {"named": 0.5, "latent": 0.2, "global": 0.3},
        }
    if name == "gradient_admission.json":
        return {
            **base,
            "cosine": {"action_reason": 0.1},
            "projection": {"rate": 0.2},
            "admission": {"rate": 0.8},
            "caps": {"slot": 0.1},
            "ema": {"slot": 0.02},
        }
    if name == "pu_stats.json":
        return {
            **base,
            "labels": [
                {
                    "label_id": label_id,
                    "gate": False,
                    "score": 0.0,
                    "lambda": 0.0,
                    "soft_positive_count": 0,
                }
                for label_id in range(21)
            ],
        }
    if name == "counterfactual.json":
        return {
            **base,
            "sample_ids": [f"fixed-{index}" for index in range(128)],
            "selected": {"effect": 0.1},
            "control": {"effect": 0.02},
            "wrong": {"effect": -0.01},
            "valid_action_target_count": 4,
            "valid_reason_target_count": 11,
        }
    if name == "calibration.json":
        return {
            **base,
            "candidates": [
                {"name": candidate, "joint": 0.4 + index * 0.001}
                for index, candidate in enumerate(
                    ("global", "group", "shrinkage_per_label", "temperature")
                )
            ],
            "chosen_thresholds": {"action": [0.0] * 4, "reason": [0.0] * 21},
            "temperature": {"action": 1.0, "reason": 1.0},
            "threshold_rms": {"action": 0.0, "reason": 0.0},
            "raw_map": {"action": 0.7, "reason": 0.5},
            "deploy_map": {"action": 0.7, "reason": 0.5},
            "fallback": {"used": False, "reason": "none"},
        }
    raise AssertionError(name)


def _epoch_payload(*, epoch: int = 3, sample_count: int = 3) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name in EPOCH_FILES:
        if name in {"logits_raw.pt", "logits_deploy.pt"}:
            payload[name] = {
                "_meta": _provenance(
                    producer=f"fate_oia.engine.eval_acpr_rael_oia:{name}",
                    epoch=epoch,
                    sample_count=sample_count,
                ),
                "action": torch.randn(sample_count, 4, dtype=torch.float32, requires_grad=True),
                "reason": torch.randn(sample_count, 21, dtype=torch.float32),
            }
        elif name == "labels.pt":
            payload[name] = {
                "_meta": _provenance(
                    producer="fate_oia.engine.eval_acpr_rael_oia:labels",
                    epoch=epoch,
                    sample_count=sample_count,
                ),
                "action": torch.zeros(sample_count, 4, dtype=torch.uint8),
                "reason": torch.zeros(sample_count, 21, dtype=torch.float32),
                "file_names": [f"sample-{index}.jpg" for index in range(sample_count)],
            }
        elif name.endswith(".jsonl"):
            case_data = (
                {
                    "labels": {"action": [1, 0, 0, 0], "reason": [0] * 21},
                    "raw_predictions": {"action": [1, 0, 0, 0], "reason": [0] * 21},
                    "deploy_predictions": {"action": [1, 0, 0, 0], "reason": [0] * 21},
                    "branch_deltas": {"full_minus_global": 0.01},
                }
                if name == "failure_cases.jsonl"
                else {
                    "target": {"type": "action", "id": 0},
                    "selected_slots": [0],
                    "masks": {"slot_0": [[1.0]]},
                    "attributes": {"slot_0": {"presence": 0.9}},
                    "contributions": {"slot_0": 0.1},
                }
            )
            payload[name] = [
                {
                    **_provenance(
                        producer=f"fate_oia.engine.export_rael_cases:{name}",
                        epoch=epoch,
                        sample_count=sample_count,
                    ),
                    "file_name": "sample-0.jpg",
                    "case_id": "sample-0",
                    "data": case_data,
                }
            ]
        else:
            payload[name] = _json_epoch_payload(
                name,
                epoch=epoch,
                sample_count=sample_count,
            )
    return payload


def _run_jsonl_row(name: str, *, epoch: int = 0, microbatch_step: int = 1):
    base = {
        **_provenance(producer=f"fate_oia.engine.train_acpr_rael_oia:{name}"),
        "epoch": epoch,
    }
    if name == "pu_audit.jsonl":
        return {
            **base,
            "label_id": microbatch_step - 1,
            "positive_count": 4,
            "baseline_auprc": 0.2,
            "pu_auprc": 0.21,
            "delta": 0.01,
            "lcb95": -0.01,
            "lambda": 0.0,
            "decision": "off",
        }
    if name == "metrics_summary.jsonl":
        return {
            **base,
            "raw_action": {"mF1": 0.6, "oF1": 0.7, "mAP": 0.72, "AUC": 0.75},
            "raw_reason": {"mF1": 0.3, "oF1": 0.4, "mAP": 0.5, "AUC": 0.6},
            "raw_joint": 0.4,
            "deploy_action": {"mF1": 0.61, "oF1": 0.71, "mAP": 0.72, "AUC": 0.75},
            "deploy_reason": {"mF1": 0.31, "oF1": 0.41, "mAP": 0.5, "AUC": 0.6},
            "deploy_joint": 0.41,
            "best_flags": {"deploy_joint": False, "action_mf1": False},
            "is_best": False,
        }
    if name == "runtime_steps.jsonl":
        return {
            **base,
            "candidate": "b4a8",
            "microbatch_step": microbatch_step,
            "optimizer_step": microbatch_step,
            "data_time": 0.1,
            "dino_time": 0.2,
            "step_time": 0.5,
            "allocated_gb": 12.0,
            "reserved_gb": 14.0,
            "dino_call_count": 1,
            "mechanism_flags": {"ledger": True, "pairwise": True, "counterfactual": True},
            "samples_per_sec": 1.25,
        }
    if name == "loss_components.jsonl":
        return {
            **base,
            "microbatch_step": microbatch_step,
            "optimizer_step": microbatch_step,
            "total_optimizer_updates": 100,
            "action": 0.4,
            "reason": 0.6,
            "grounding": 0.2,
            "pairwise_auxiliary": 0.1,
            "counterfactual": 0.05,
            "non_regression": 0.02,
            "feature_view": 0.01,
            "pu_private": 0.0,
            "grounding_weighted": 0.01,
            "pairwise_auxiliary_weighted": 0.005,
            "counterfactual_weighted": 0.0025,
            "non_regression_weighted": 0.0004,
            "feature_view_weighted": 0.0002,
            "r5": min(1.0, microbatch_step / 5.0),
            "r10": min(1.0, microbatch_step / 10.0),
            "total": 1.0185,
            "valid_counts": {"grounding": 4, "counterfactual": 2},
        }
    if name == "gradient_admission.jsonl":
        return {
            **base,
            "microbatch_step": microbatch_step,
            "optimizer_step": microbatch_step,
            "raw_norms": {"slot": 0.2},
            "projected_norms": {"slot": 0.1},
            "cosines": {"action_reason": -0.1},
            "caps": {"slot": 0.1},
            "ema_norms": {"slot": 0.08},
            "registered": 2,
            "triggered": 1,
            "removed": 1,
        }
    if name == "mechanism_stats.jsonl":
        base["producer"] = "fate_oia.engine.train_acpr_rael_oia:mechanism_stats"
        row = {
            **base,
            "microbatch_step": microbatch_step,
            "optimizer_step": microbatch_step,
        }
        scalar_fields = (
            "data_time",
            "dino_time",
            "field_time",
            "slot_time",
            "category_time",
            "relation_time",
            "backward_time",
            "optimizer_time",
            "samples_per_sec",
            "allocated_gb",
            "reserved_gb",
            "action_global_loss",
            "action_final_loss",
            "reason_global_loss",
            "reason_final_loss",
            "action_global_logit_rms",
            "reason_global_logit_rms",
            "action_unary_rms_over_global",
            "action_pairwise_rms_over_global",
            "reason_unary_rms_over_global",
            "reason_pairwise_rms_over_global",
            "gamma_AS",
            "gamma_RA",
            "gamma_unary",
            "gamma_pairwise",
            "active_entity_count",
            "background_mass",
            "latent_mass",
            "slot_mask_entropy",
            "slot_pair_iou",
            "slot_area_mean",
            "slot_area_std",
            "entity_type_entropy",
            "traffic_state_entropy",
            "road_coverage",
            "named_contribution_ratio",
            "latent_contribution_ratio",
            "global_contribution_ratio",
            "layer_entropy",
            "positive_weight_mean",
            "negative_weight_mean",
            "pu_active_label_count",
            "pu_soft_positive_count",
            "semantic_private_norm_ratio",
            "action_reason_context_norm",
            "analytic_selected_effect",
            "feature_selected_effect",
            "control_effect",
            "wrong_target_effect",
            "sign_consistency",
            "valid_action_target_count",
            "valid_reason_target_count",
        )
        row.update({field: 0.1 for field in scalar_fields})
        row["dino_call_count"] = 1
        row["optimizer_stepped"] = True
        row["owner_gradient_norms"] = {owner: 0.1 for owner in OWNER_NAMES}
        row["owner_parameter_delta"] = {
            owner: 0.001 if owner == "action_category" else 0.0
            for owner in OWNER_NAMES
        }
        for field in (
            "action_layer_weights",
            "reason_layer_weights",
            "slot_layer_weights",
            "layer_collapse",
            "slot_cos_action_reason",
            "slot_cos_action_grounding",
            "slot_cos_action_cf",
            "negative_rates",
            "projection_rates",
            "admission_rates",
            "raw_norms",
            "projected_norms",
            "budget_hit_rates",
            "ema_norms",
            "pu_lambda_by_label",
        ):
            row[field] = {"mean": 0.1}
        return row
    return {
        **base,
        "microbatch_step": microbatch_step,
        "optimizer_step": microbatch_step,
        "value": 0.25,
    }


def _write_complete_run_root(module, root: Path) -> object:
    writer = module.RAELArtifactWriter(root)
    immutable = ("source_fingerprint.json", "config_resolved.yaml")
    for name in immutable:
        writer.write_run_file(name, _run_payload(name))
    for name in sorted(RUN_ROOT_FILES.difference(immutable)):
        if name.endswith(".jsonl"):
            writer.append_run_jsonl(name, _run_jsonl_row(name))
        else:
            writer.write_run_file(name, _run_payload(name))
    return writer


def _attach_public_optimizer_fixture(trainer: object) -> None:
    model = torch.nn.Module()
    groups: list[dict[str, object]] = []
    for owner in OWNER_NAMES:
        head = torch.nn.Linear(1, 1)
        model.add_module(owner, head)
        lr = 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002
        groups.extend(
            (
                {
                    "params": [head.weight],
                    "lr": lr,
                    "weight_decay": 0.05,
                    "owner": owner,
                },
                {
                    "params": [head.bias],
                    "lr": lr,
                    "weight_decay": 0.0,
                    "owner": owner,
                },
            )
        )
    trainer.model = model
    trainer.optimizer_bundle = SimpleNamespace(
        optimizer=torch.optim.SGD(groups),
        no_decay_parameter_names=tuple(f"{owner}.bias" for owner in OWNER_NAMES),
    )


def test_p18_declares_atomic_contract_file_sets() -> None:
    module = _module()
    assert set(module.RUN_ROOT_FILES) == RUN_ROOT_FILES
    assert set(module.EPOCH_FILES) == EPOCH_FILES
    assert tuple(module._BRANCH_NAMES) == tuple(MODEL_BRANCH_NAMES)


def test_p18_writes_complete_run_root_and_epoch_transactionally(tmp_path: Path) -> None:
    module = _module()
    writer = module.RAELArtifactWriter(tmp_path)
    records = []
    for name in sorted(RUN_ROOT_FILES):
        if name.endswith(".jsonl"):
            records += writer.append_run_jsonl(name, _run_jsonl_row(name))
        else:
            records += writer.write_run_file(name, _run_payload(name))
    records += writer.write_epoch(3, _epoch_payload())

    assert {path.name for path in tmp_path.iterdir() if path.is_file()} == RUN_ROOT_FILES
    epoch_dir = tmp_path / "epoch_003"
    assert {path.name for path in epoch_dir.iterdir()} == EPOCH_FILES
    assert not list(tmp_path.glob(".epoch_003.staging-*"))
    assert yaml.safe_load((tmp_path / "config_resolved.yaml").read_text(encoding="utf-8"))[
        "resolved_config"
    ]["training"]["epochs"] == 14
    stored = torch.load(epoch_dir / "logits_raw.pt", map_location="cpu", weights_only=True)
    assert stored["action"].shape == (3, 4)
    assert stored["reason"].shape == (3, 21)
    assert stored["action"].device.type == "cpu"
    assert stored["action"].requires_grad is False
    assert len(records) == len(RUN_ROOT_FILES) + len(EPOCH_FILES)
    for record in records:
        target = tmp_path / record["relative_path"]
        assert target.is_file()
        assert record["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
        assert record["bytes"] == target.stat().st_size


def test_p18_epoch_transaction_rolls_back_on_injected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    writer = module.RAELArtifactWriter(tmp_path)
    original = module._write_staged_file
    calls = 0

    def fail_after_three(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected staging failure")
        original(path, content)

    monkeypatch.setattr(module, "_write_staged_file", fail_after_three)
    with pytest.raises(OSError, match="injected"):
        writer.write_epoch(3, _epoch_payload())
    assert not (tmp_path / "epoch_003").exists()
    assert not list(tmp_path.glob(".epoch_003.staging-*"))


def test_p18_jsonl_is_locked_validated_and_monotonic(tmp_path: Path) -> None:
    module = _module()
    writer = module.RAELArtifactWriter(tmp_path)
    writer.append_run_jsonl(
        "loss_components.jsonl",
        _run_jsonl_row("loss_components.jsonl", microbatch_step=1),
    )
    writer.append_run_jsonl(
        "loss_components.jsonl",
        _run_jsonl_row("loss_components.jsonl", microbatch_step=2),
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "loss_components.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["microbatch_step"] for row in rows] == [1, 2]
    with pytest.raises(ValueError, match="monotonic"):
        writer.append_run_jsonl(
            "loss_components.jsonl",
            _run_jsonl_row("loss_components.jsonl", microbatch_step=2),
        )
    with pytest.raises(ValueError, match="finite"):
        bad = _run_jsonl_row("loss_components.jsonl", microbatch_step=3)
        bad["value"] = float("nan")
        writer.append_run_jsonl("loss_components.jsonl", bad)
    assert len((tmp_path / "loss_components.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    (tmp_path / "mechanism_stats.jsonl").write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="existing JSONL"):
        writer.append_run_jsonl(
            "mechanism_stats.jsonl",
            _run_jsonl_row("mechanism_stats.jsonl"),
        )


def test_p18_jsonl_concurrent_append_does_not_lose_rows(tmp_path: Path) -> None:
    module = _module()
    writer = module.RAELArtifactWriter(tmp_path)

    def append(label_id: int) -> None:
        writer.append_run_jsonl(
            "pu_audit.jsonl",
            _run_jsonl_row("pu_audit.jsonl", epoch=0, microbatch_step=label_id + 1),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(21)))
    rows = [
        json.loads(line)
        for line in (tmp_path / "pu_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 21
    assert {row["label_id"] for row in rows} == set(range(21))


def test_p18_rejects_partial_extra_unsafe_and_mutating_epoch_writes(tmp_path: Path) -> None:
    module = _module()
    writer = module.RAELArtifactWriter(tmp_path)
    payload = _epoch_payload()
    missing = dict(payload)
    missing.pop("pu_stats.json")
    with pytest.raises(ValueError, match="exactly"):
        writer.write_epoch(3, missing)
    extra = dict(payload)
    extra["unexpected.json"] = {}
    with pytest.raises(ValueError, match="exactly"):
        writer.write_epoch(3, extra)
    with pytest.raises(ValueError, match="allowed"):
        writer.write_run_file("../escape.json", {})
    with pytest.raises(ValueError, match="finite"):
        bad = _run_payload("run_manifest.json")
        bad["seed"] = float("inf")
        writer.write_run_file("run_manifest.json", bad)

    writer.write_epoch(3, payload)
    writer.write_epoch(3, payload)
    changed = _epoch_payload()
    changed["raw_metrics.json"]["metrics"]["joint"] = 0.9
    with pytest.raises(FileExistsError, match="immutable"):
        writer.write_epoch(3, changed)


@pytest.mark.parametrize(
    ("name", "mutator", "message"),
    [
        (
            "logits_raw.pt",
            lambda payload: payload["logits_raw.pt"].update(
                action=torch.zeros(3, 5, dtype=torch.float32)
            ),
            "shape",
        ),
        (
            "logits_deploy.pt",
            lambda payload: payload["logits_deploy.pt"].update(
                reason=torch.zeros(3, 21, dtype=torch.float64)
            ),
            "float32",
        ),
        (
            "labels.pt",
            lambda payload: payload["labels.pt"].update(
                reason=torch.full((3, 21), 0.5, dtype=torch.float32)
            ),
            "binary",
        ),
        (
            "labels.pt",
            lambda payload: payload["labels.pt"].update(
                file_names=["only-one.jpg"]
            ),
            "sample_count",
        ),
        (
            "logits_raw.pt",
            lambda payload: payload["logits_raw.pt"].update(
                action=torch.ones(3, 4, dtype=torch.complex64)
            ),
            "complex",
        ),
    ],
)
def test_p18_rejects_wrong_tensor_contract(
    tmp_path: Path,
    name: str,
    mutator,
    message: str,
) -> None:
    module = _module()
    payload = _epoch_payload()
    mutator(payload)
    with pytest.raises((TypeError, ValueError), match=message):
        module.RAELArtifactWriter(tmp_path).write_epoch(3, payload)


def test_p18_rejects_missing_provenance_and_placeholder_json(tmp_path: Path) -> None:
    module = _module()
    payload = _epoch_payload()
    payload["slot_stats.json"].pop("producer")
    with pytest.raises(ValueError, match="producer"):
        module.RAELArtifactWriter(tmp_path).write_epoch(3, payload)
    run_payload = _run_payload("runtime_profile.json")
    run_payload["candidates"] = []
    with pytest.raises(ValueError, match="nonempty"):
        module.RAELArtifactWriter(tmp_path).write_run_file(
            "runtime_profile.json",
            run_payload,
        )


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        ("slot_stats.json", "reliability"),
        ("layer_stats.json", "slot_layer_weights"),
        ("relation_stats.json", "total_pair_count"),
        ("contribution_stats.json", "reconstruction_error"),
        ("named_latent_global.json", "per_target"),
        ("gradient_admission.json", "ema"),
        ("pu_stats.json", "labels"),
        ("counterfactual.json", "sample_ids"),
        ("calibration.json", "candidates"),
    ],
)
def test_p18_rejects_missing_exact_epoch_schema(
    tmp_path: Path,
    artifact: str,
    field: str,
) -> None:
    module = _module()
    payload = _epoch_payload()
    payload[artifact].pop(field)
    with pytest.raises(ValueError, match=field):
        module.RAELArtifactWriter(tmp_path).write_epoch(3, payload)


def test_p18_rejects_wrong_branch_names_and_incomplete_branch_metrics(tmp_path: Path) -> None:
    module = _module()
    payload = _epoch_payload()
    payload["branch_metrics.json"]["branches"][0]["name"] = "invented_branch"
    with pytest.raises(ValueError, match="branch name"):
        module.RAELArtifactWriter(tmp_path).write_epoch(3, payload)
    payload = _epoch_payload()
    payload["branch_metrics.json"]["branches"][0]["metrics"].pop("reason")
    with pytest.raises((TypeError, ValueError), match="reason"):
        module.RAELArtifactWriter(tmp_path).write_epoch(3, payload)


def test_p18_rejects_zero_sample_epoch_and_placeholder_mechanism_rows(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="sample_count"):
        module.RAELArtifactWriter(tmp_path / "zero").write_epoch(
            3,
            _epoch_payload(sample_count=0),
        )

    row = _run_jsonl_row("mechanism_stats.jsonl")
    row["dino_call_count"] = 0
    with pytest.raises(ValueError, match="dino_call_count"):
        module.RAELArtifactWriter(tmp_path / "calls").append_run_jsonl(
            "mechanism_stats.jsonl",
            row,
        )
    row = _run_jsonl_row("mechanism_stats.jsonl")
    row["data_time"] = -0.1
    with pytest.raises(ValueError, match="data_time"):
        module.RAELArtifactWriter(tmp_path / "timing").append_run_jsonl(
            "mechanism_stats.jsonl",
            row,
        )


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        ("runtime_steps.jsonl", "mechanism_flags"),
        ("loss_components.jsonl", "valid_counts"),
        ("gradient_admission.jsonl", "projected_norms"),
        ("mechanism_stats.jsonl", "action_global_loss"),
        ("metrics_summary.jsonl", "deploy_reason"),
        ("pu_audit.jsonl", "lcb95"),
    ],
)
def test_p18_rejects_missing_exact_run_jsonl_schema(
    tmp_path: Path,
    artifact: str,
    field: str,
) -> None:
    module = _module()
    row = _run_jsonl_row(artifact)
    row.pop(field)
    with pytest.raises(ValueError, match=field):
        module.RAELArtifactWriter(tmp_path).append_run_jsonl(artifact, row)


def test_p18_rejects_symlinked_artifact_path(tmp_path: Path) -> None:
    module = _module()
    target = tmp_path / "outside"
    target.mkdir()
    linked_root = tmp_path / "linked-run"
    try:
        linked_root.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(ValueError, match="symlink"):
        module.RAELArtifactWriter(linked_root)


def test_p18_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    module = _module()
    outside = tmp_path / "outside-parent"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    child = linked_parent / "run"
    child.mkdir()
    with pytest.raises(ValueError, match="symlink"):
        module.RAELArtifactWriter(child)


def test_p18_revalidates_root_after_construction(tmp_path: Path) -> None:
    module = _module()
    run_root = tmp_path / "run"
    writer = module.RAELArtifactWriter(run_root)
    run_root.rmdir()
    outside = tmp_path / "outside-after-construction"
    outside.mkdir()
    try:
        run_root.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(ValueError, match="symlink"):
        writer.write_run_file("run_manifest.json", _run_payload("run_manifest.json"))


def test_p18_lock_does_not_require_windows_o_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.delattr(module.os, "O_BINARY", raising=False)
    module.RAELArtifactWriter(tmp_path).append_run_jsonl(
        "loss_components.jsonl",
        _run_jsonl_row("loss_components.jsonl"),
    )


def test_p18_rejects_unbound_fingerprint_and_invalid_owner_manifest(tmp_path: Path) -> None:
    module = _module()
    fingerprint = _run_payload("source_fingerprint.json")
    fingerprint["file_sha256"] = {}
    with pytest.raises(ValueError, match="file_sha256"):
        module.RAELArtifactWriter(tmp_path / "fingerprint").write_run_file(
            "source_fingerprint.json",
            fingerprint,
        )

    owners = _run_payload("optimizer_owners.json")
    owners["owners"]["multilayer_field"]["count"] = 3
    with pytest.raises(ValueError, match="count"):
        module.RAELArtifactWriter(tmp_path / "owners").write_run_file(
            "optimizer_owners.json",
            owners,
        )


def test_p18_requires_exact_p17_v4_six_group_fingerprint_recomputation(
    tmp_path: Path,
) -> None:
    module = _module()
    fingerprint = _run_payload("source_fingerprint.json")
    assert tuple(fingerprint["groups"]) == P17_FINGERPRINT_GROUPS
    assert set(fingerprint["groups"]) == set(P17_FINGERPRINT_GROUPS)
    assert fingerprint["group_hashes"] == P17_FINGERPRINT["group_hashes"]
    assert fingerprint["source_hash"] == P17_FINGERPRINT["source_hash"]
    assert fingerprint["config_hash"] == P17_FINGERPRINT["config_hash"]
    assert fingerprint["schema_hash"] == P17_FINGERPRINT["schema_hash"]
    assert fingerprint["required_files_hash"] == P17_FINGERPRINT["required_files_hash"]

    module.RAELArtifactWriter(tmp_path / "valid").write_run_file(
        "source_fingerprint.json",
        fingerprint,
    )

    mutations = {
        "group": lambda value: value["group_hashes"].__setitem__("source", "0" * 64),
        "source": lambda value: value.__setitem__("source_hash", "0" * 64),
        "config": lambda value: value.__setitem__("config_hash", "0" * 64),
        "schema": lambda value: value.__setitem__("schema_hash", "0" * 64),
        "required": lambda value: (
            value.__setitem__("required_files_hash", "0" * 64),
            value.__setitem__("source_fingerprint_sha256", "0" * 64),
        ),
    }
    for name, mutate in mutations.items():
        invalid = copy.deepcopy(fingerprint)
        mutate(invalid)
        with pytest.raises(ValueError, match="fingerprint|hash"):
            module.RAELArtifactWriter(tmp_path / name).write_run_file(
                "source_fingerprint.json",
                invalid,
            )


def test_p18_rejects_branch_metrics_without_per_label_action_and_reason_rows(
    tmp_path: Path,
) -> None:
    module = _module()
    payload = _epoch_payload()
    for branch in payload["branch_metrics.json"]["branches"]:
        assert len(branch["per_action"]) == 4
        assert len(branch["per_reason"]) == 21

    missing_action = _epoch_payload()
    missing_action["branch_metrics.json"]["branches"][0].pop("per_action")
    with pytest.raises(ValueError, match="per_action"):
        module.RAELArtifactWriter(tmp_path / "missing-action").write_epoch(3, missing_action)

    missing_reason = _epoch_payload()
    missing_reason["branch_metrics.json"]["branches"][0]["per_reason"] = _per_label_rows(20)
    with pytest.raises(ValueError, match="per_reason"):
        module.RAELArtifactWriter(tmp_path / "missing-reason").write_epoch(3, missing_reason)


def test_p18_rejects_non_single_dino_and_unattributed_mechanism_rows(
    tmp_path: Path,
) -> None:
    module = _module()

    extra_dino = _run_jsonl_row("mechanism_stats.jsonl")
    extra_dino["dino_call_count"] = 2
    with pytest.raises(ValueError, match="dino_call_count"):
        module.RAELArtifactWriter(tmp_path / "extra-dino").append_run_jsonl(
            "mechanism_stats.jsonl",
            extra_dino,
        )

    missing_gradients = _run_jsonl_row("mechanism_stats.jsonl")
    missing_gradients.pop("owner_gradient_norms")
    with pytest.raises(ValueError, match="owner_gradient_norms"):
        module.RAELArtifactWriter(tmp_path / "missing-gradients").append_run_jsonl(
            "mechanism_stats.jsonl",
            missing_gradients,
        )

    stepped_without_delta = _run_jsonl_row("mechanism_stats.jsonl")
    stepped_without_delta["owner_parameter_delta"] = {
        owner: 0.0 for owner in OWNER_NAMES
    }
    with pytest.raises(ValueError, match="parameter_delta|optimizer_stepped"):
        module.RAELArtifactWriter(tmp_path / "zero-delta").append_run_jsonl(
            "mechanism_stats.jsonl",
            stepped_without_delta,
        )


def test_p18_uses_strict_public_trainer_whitelists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        lambda *args, **kwargs: copy.deepcopy(P17_FINGERPRINT),
    )

    class PublicTrainer:
        def __init__(self) -> None:
            _attach_public_optimizer_fixture(self)

        def state_dict(self) -> dict[str, object]:
            owners = {
                owner: (f"{owner}.weight",)
                for owner in OWNER_NAMES
            }
            owners = {
                owner: (f"{owner}.weight", f"{owner}.bias")
                for owner in OWNER_NAMES
            }
            fingerprint = {
                key: value
                for key, value in _run_payload("source_fingerprint.json").items()
                if key
                not in {
                    "schema_version",
                    "producer",
                    "source_fingerprint_sha256",
                    "config_sha256",
                }
            }
            fingerprint["private_nested"] = {"secret": object()}
            return {
                "checkpoint_schema": "rael-p17-resume-v3",
                "owner_parameter_names": owners,
                "trainer_config": {
                    "precision": "bf16",
                    "gradient_accumulation_steps": 8,
                    "total_optimizer_updates": 100,
                    "owner_learning_rates": {
                        owner: 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002
                        for owner in owners
                    },
                    "private_nested": {"secret": object()},
                },
                "resume_fingerprints": fingerprint,
                "private_payload_that_must_not_escape": object(),
            }

    trainer = PublicTrainer()
    with pytest.raises((TypeError, ValueError), match="artifact_context"):
        module.trainer_run_artifact_contract(trainer)

    contract = module.trainer_run_artifact_contract(
        trainer,
        artifact_context=_provenance(
            producer="fate_oia.engine.train_acpr_rael_oia:trainer_contract"
        ),
    )
    assert set(contract) == {"source_fingerprint", "optimizer_owners"}
    assert contract["source_fingerprint"]["phase"] == "development"
    all_parameter_names: set[str] = set()
    owners = contract["optimizer_owners"]["owners"]
    assert tuple(owners) == OWNER_NAMES
    for owner in OWNER_NAMES:
        manifest = owners[owner]
        expected_lr = 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002
        assert manifest["lr"] == pytest.approx(expected_lr)
        assert manifest["count"] == len(manifest["parameter_names"])
        assert len(manifest["parameters"]) == manifest["count"]
        assert {parameter["name"] for parameter in manifest["parameters"]} == set(
            manifest["parameter_names"]
        )
        assert all(parameter["lr"] == pytest.approx(expected_lr) for parameter in manifest["parameters"])
        assert {parameter["weight_decay"] for parameter in manifest["parameters"]} <= {0.0, 0.05}
        assert not all_parameter_names.intersection(manifest["parameter_names"])
        all_parameter_names.update(manifest["parameter_names"])
    assert len(all_parameter_names) == sum(owner["count"] for owner in owners.values())
    encoded = json.dumps(contract, allow_nan=False)
    assert "private_nested" not in encoded
    assert "secret" not in encoded

    writer = module.RAELArtifactWriter(tmp_path / "public-contract")
    writer.write_run_file("source_fingerprint.json", contract["source_fingerprint"])
    writer.write_run_file("optimizer_owners.json", contract["optimizer_owners"])

    class UnknownOwnerTrainer(PublicTrainer):
        def state_dict(self) -> dict[str, object]:
            state = super().state_dict()
            state["owner_parameter_names"]["unexpected_owner"] = ("unexpected_owner.weight",)
            state["trainer_config"]["owner_learning_rates"]["unexpected_owner"] = 0.0002
            return state

    with pytest.raises(ValueError, match="owner topology"):
        module.trainer_run_artifact_contract(
            UnknownOwnerTrainer(),
            artifact_context=_provenance(
                producer="fate_oia.engine.train_acpr_rael_oia:trainer_contract"
            ),
        )

    step = SimpleNamespace(
        components={
            "action": torch.tensor(0.4, requires_grad=True),
            "reason": torch.tensor(0.6),
        },
        optimizer_stepped=True,
        optimizer_step=8,
        microbatch_step=64,
        owner_gradient_norms_pre_clip={"action_category": 1.2},
        owner_gradient_norms_post_clip={"action_category": 0.8},
        owner_task_gradient_norms_pre_clip={"action_category": 1.1},
        owner_parameter_delta={"action_category": 0.01},
        owner_optimizer_effect_delta={"action_category": 0.009},
        owner_decay_only_parameter_delta={"action_category": 0.001},
        owner_optimizer_step_count={"action_category": 8},
        admission_registered_count=2,
        admission_triggered_count=2,
        admission_removed_count=2,
    )
    with pytest.raises((TypeError, ValueError), match="artifact_context"):
        module.step_result_artifact_rows(step)

    formal_components = {
        "action": torch.tensor(0.4),
        "reason": torch.tensor(0.6),
        "grounding": torch.tensor(0.2),
        "pairwise_auxiliary": torch.tensor(0.1),
        "counterfactual": torch.tensor(0.05),
        "non_regression": torch.tensor(0.02),
        "feature_view": torch.tensor(0.01),
        "pu_private": torch.tensor(0.0),
        "grounding_weighted": torch.tensor(0.01),
        "pairwise_auxiliary_weighted": torch.tensor(0.005),
        "counterfactual_weighted": torch.tensor(0.0025),
        "non_regression_weighted": torch.tensor(0.0004),
        "feature_view_weighted": torch.tensor(0.0002),
        "total": torch.tensor(1.0181),
    }
    formal_step = SimpleNamespace(**{**vars(step), "components": formal_components})
    complete_rows = module.step_result_artifact_rows(
        formal_step,
        artifact_context={
            **_provenance(producer="fate_oia.engine.train_acpr_rael_oia:step"),
            "epoch": 0,
            "total_optimizer_updates": 100,
            "valid_counts": {"grounding": 4, "counterfactual": 2},
            "admission": {
                "raw_norms": {"slot": 0.2},
                "projected_norms": {"slot": 0.1},
                "cosines": {"action_reason": -0.1},
                "caps": {"slot": 0.1},
                "ema_norms": {"slot": 0.08},
            },
        },
    )
    assert complete_rows["loss_components"]["r5"] == pytest.approx(1.0)
    assert complete_rows["loss_components"]["r10"] == pytest.approx(0.8)
    writer = module.RAELArtifactWriter(tmp_path / "public-rows")
    writer.append_run_jsonl(
        "loss_components.jsonl",
        complete_rows["loss_components"],
    )
    writer.append_run_jsonl(
        "gradient_admission.jsonl",
        complete_rows["gradient_admission"],
    )


def test_p18_reopens_complete_root_after_sorted_fingerprint_persistence(
    tmp_path: Path,
) -> None:
    """A JSON sort_keys rewrite must not invalidate an otherwise exact P17 fingerprint."""

    module = _module()
    root = tmp_path / "sorted-fingerprint-root"
    _write_complete_run_root(module, root)

    records = module.RAELArtifactWriter(root).validate_run_root_complete()
    assert {record["relative_path"] for record in records} == RUN_ROOT_FILES


def test_p18_fails_closed_on_cross_writer_root_and_epoch_provenance(
    tmp_path: Path,
) -> None:
    """Every durable artifact must bind one run's source/config identities."""

    module = _module()

    invalid_config = _run_payload("config_resolved.yaml")
    invalid_config["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="config_sha256|resolved_config"):
        module.RAELArtifactWriter(tmp_path / "invalid-config").write_run_file(
            "config_resolved.yaml",
            invalid_config,
        )

    root = tmp_path / "cross-writer"
    _write_complete_run_root(module, root)
    reopened = module.RAELArtifactWriter(root)
    foreign_row = _run_jsonl_row("loss_components.jsonl", microbatch_step=2)
    foreign_row["source_fingerprint_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="provenance|source_fingerprint"):
        reopened.append_run_jsonl("loss_components.jsonl", foreign_row)

    foreign_epoch = _epoch_payload()
    foreign_epoch["raw_metrics.json"]["config_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="provenance|config_sha256"):
        reopened.write_epoch(3, foreign_epoch)


def test_p18_trainer_contract_recomputes_fingerprint_with_p17_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A state_dict fingerprint is evidence to compare, never a trusted replacement."""

    module = _module()

    class FingerprintTrainer:
        def __init__(self) -> None:
            _attach_public_optimizer_fixture(self)

        def state_dict(self) -> dict[str, object]:
            return {
                "checkpoint_schema": "rael-p17-resume-v3",
                "owner_parameter_names": {
                    owner: (f"{owner}.weight", f"{owner}.bias")
                    for owner in OWNER_NAMES
                },
                "trainer_config": {
                    "precision": "bf16",
                    "gradient_accumulation_steps": 8,
                    "total_optimizer_updates": 100,
                    "owner_learning_rates": {
                        owner: 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002
                        for owner in OWNER_NAMES
                    },
                },
                "resume_fingerprints": copy.deepcopy(P17_FINGERPRINT),
            }

    trainer = FingerprintTrainer()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def matching_builder(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append((args, dict(kwargs)))
        return copy.deepcopy(P17_FINGERPRINT)

    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        matching_builder,
        raising=False,
    )
    module.trainer_run_artifact_contract(
        trainer,
        artifact_context=_provenance(producer="fate_oia.engine.train_acpr_rael_oia:contract"),
    )
    assert calls, "contract must call the live P17 fingerprint builder"

    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        lambda *args, **kwargs: _p17_fingerprint_with_source_change(),
    )
    with pytest.raises(ValueError, match="fingerprint|recompute|mismatch"):
        module.trainer_run_artifact_contract(
            trainer,
            artifact_context=_provenance(producer="fate_oia.engine.train_acpr_rael_oia:contract"),
        )


def test_p18_owner_contract_matches_real_optimizer_parameter_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persisted owner table must be derived from actual optimizer param_groups."""

    module = _module()
    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        lambda *args, **kwargs: copy.deepcopy(P17_FINGERPRINT),
    )

    class OptimizerBackedTrainer:
        def __init__(self) -> None:
            self.model = torch.nn.Module()
            groups = []
            self.owner_parameter_names: dict[str, tuple[str, str]] = {}
            for owner in OWNER_NAMES:
                head = torch.nn.Linear(1, 1)
                self.model.add_module(owner, head)
                names = (f"{owner}.weight", f"{owner}.bias")
                self.owner_parameter_names[owner] = names
                lr = 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002
                groups.extend(
                    (
                        {"params": [head.weight], "lr": lr, "weight_decay": 0.05},
                        {"params": [head.bias], "lr": lr, "weight_decay": 0.0},
                    )
                )
            self.optimizer_bundle = SimpleNamespace(
                optimizer=torch.optim.SGD(groups),
                no_decay_parameter_names=tuple(f"{owner}.bias" for owner in OWNER_NAMES),
            )

        def state_dict(self) -> dict[str, object]:
            return {
                "checkpoint_schema": "rael-p17-resume-v3",
                "owner_parameter_names": self.owner_parameter_names,
                "trainer_config": {
                    "precision": "bf16",
                    "gradient_accumulation_steps": 8,
                    "total_optimizer_updates": 100,
                    "owner_learning_rates": {
                        owner: 0.0003 if owner in PRIVATE_OWNER_NAMES else 0.0002
                        for owner in OWNER_NAMES
                    },
                },
                "resume_fingerprints": copy.deepcopy(P17_FINGERPRINT),
            }

    trainer = OptimizerBackedTrainer()
    contract = module.trainer_run_artifact_contract(
        trainer,
        artifact_context=_provenance(producer="fate_oia.engine.train_acpr_rael_oia:contract"),
    )
    named_by_object = {id(parameter): name for name, parameter in trainer.model.named_parameters()}
    expected_by_name = {
        named_by_object[id(parameter)]: (group["lr"], group["weight_decay"])
        for group in trainer.optimizer_bundle.optimizer.param_groups
        for parameter in group["params"]
    }
    for owner in OWNER_NAMES:
        for parameter in contract["optimizer_owners"]["owners"][owner]["parameters"]:
            assert expected_by_name[parameter["name"]] == pytest.approx(
                (parameter["lr"], parameter["weight_decay"])
            )

    trainer.optimizer_bundle.optimizer.param_groups[0]["lr"] = 0.123
    with pytest.raises(ValueError, match="optimizer|learning rate|LR"):
        module.trainer_run_artifact_contract(
            trainer,
            artifact_context=_provenance(producer="fate_oia.engine.train_acpr_rael_oia:contract"),
        )


def test_p18_loss_component_rows_recompute_r5_r10_in_the_validator(
    tmp_path: Path,
) -> None:
    """A caller cannot forge schedule progress after a step result is emitted."""

    module = _module()
    forged = _run_jsonl_row("loss_components.jsonl", microbatch_step=8)
    forged["optimizer_step"] = 8
    forged["total_optimizer_updates"] = 100
    forged["r5"] = 0.12
    forged["r10"] = 0.12
    with pytest.raises(ValueError, match="r5|r10|schedule"):
        module.RAELArtifactWriter(tmp_path).append_run_jsonl("loss_components.jsonl", forged)


def test_p18_mechanism_rows_require_formal_signal_and_reject_constant_traces(
    tmp_path: Path,
) -> None:
    """Mechanism telemetry must be formal, active, and non-placeholder over time."""

    module = _module()

    wrong_producer = _run_jsonl_row("mechanism_stats.jsonl")
    wrong_producer["producer"] = "fate_oia.engine.eval_acpr_rael_oia:mechanism_stats"
    with pytest.raises(ValueError, match="producer|trainer"):
        module.RAELArtifactWriter(tmp_path / "wrong-producer").append_run_jsonl(
            "mechanism_stats.jsonl",
            wrong_producer,
        )

    zero_gradients = _run_jsonl_row("mechanism_stats.jsonl")
    zero_gradients["owner_gradient_norms"] = {owner: 0.0 for owner in OWNER_NAMES}
    with pytest.raises(ValueError, match="owner_gradient_norms|nonzero"):
        module.RAELArtifactWriter(tmp_path / "zero-gradients").append_run_jsonl(
            "mechanism_stats.jsonl",
            zero_gradients,
        )

    zero_mechanism = _run_jsonl_row("mechanism_stats.jsonl")
    for field in (
        "action_global_loss",
        "action_final_loss",
        "reason_global_loss",
        "reason_final_loss",
        "gamma_AS",
        "gamma_RA",
        "named_contribution_ratio",
        "analytic_selected_effect",
        "feature_selected_effect",
    ):
        zero_mechanism[field] = 0.0
    with pytest.raises(ValueError, match="mechanism|critical|nonzero"):
        module.RAELArtifactWriter(tmp_path / "zero-mechanism").append_run_jsonl(
            "mechanism_stats.jsonl",
            zero_mechanism,
        )

    root = tmp_path / "constant-trace"
    writer = _write_complete_run_root(module, root)
    for step in (2, 3):
        writer.append_run_jsonl(
            "mechanism_stats.jsonl",
            _run_jsonl_row("mechanism_stats.jsonl", microbatch_step=step),
        )
    with pytest.raises(ValueError, match="constant|mechanism|trace"):
        module.RAELArtifactWriter(root).validate_run_root_complete()


def test_p18_jsonl_append_is_incremental_not_full_parse_or_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    writer = module.RAELArtifactWriter(tmp_path)
    writer.append_run_jsonl(
        "loss_components.jsonl",
        _run_jsonl_row("loss_components.jsonl", microbatch_step=1),
    )

    monkeypatch.setattr(
        module,
        "_parse_existing_jsonl",
        lambda *args, **kwargs: pytest.fail("append must not parse full JSONL history"),
    )
    monkeypatch.setattr(
        module,
        "_atomic_replace",
        lambda *args, **kwargs: pytest.fail("append must not rewrite full JSONL history"),
    )
    writer.append_run_jsonl(
        "loss_components.jsonl",
        _run_jsonl_row("loss_components.jsonl", microbatch_step=2),
    )

    lines = (tmp_path / "loss_components.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["microbatch_step"] for line in lines] == [1, 2]


def test_p18_cross_file_concurrent_writers_share_one_identity_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    original = module.RAELArtifactWriter._read_root_provenances_locked

    def delayed_read(self, *, require_complete: bool):
        values = original(self, require_complete=require_complete)
        time.sleep(0.05)
        return values

    monkeypatch.setattr(
        module.RAELArtifactWriter,
        "_read_root_provenances_locked",
        delayed_read,
    )
    root = tmp_path / "identity-race"
    first = _run_jsonl_row("loss_components.jsonl", microbatch_step=1)
    second = _run_jsonl_row("gradient_admission.jsonl", microbatch_step=1)
    second["source_fingerprint_sha256"] = "9" * 64

    def append(name: str, row: dict[str, object]) -> str:
        try:
            module.RAELArtifactWriter(root).append_run_jsonl(name, row)
            return "ok"
        except ValueError as error:
            assert "provenance" in str(error) or "identity" in str(error)
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: append(*item),
                (
                    ("loss_components.jsonl", first),
                    ("gradient_admission.jsonl", second),
                ),
            )
        )
    assert sorted(results) == ["ok", "rejected"]


def test_p18_rejects_uppercase_digest_even_when_fingerprint_hashes_are_consistent(
    tmp_path: Path,
) -> None:
    module = _module()
    fingerprint = copy.deepcopy(P17_FINGERPRINT)
    source_path = fingerprint["groups"]["source"][0]
    fingerprint["file_sha256"][source_path] = "A" * 64
    _recompute_p17_fingerprint(fingerprint)
    payload = {
        **_provenance(producer="fate_oia.tests.uppercase-fingerprint"),
        **fingerprint,
        "source_fingerprint_sha256": fingerprint["required_files_hash"],
    }
    with pytest.raises(ValueError, match="SHA256|lowercase|digest"):
        module.RAELArtifactWriter(tmp_path).write_run_file(
            "source_fingerprint.json",
            payload,
        )


def test_p18_reopened_jsonl_without_final_newline_preserves_row_separator(
    tmp_path: Path,
) -> None:
    module = _module()
    first = _run_jsonl_row("loss_components.jsonl", microbatch_step=1)
    path = tmp_path / "loss_components.jsonl"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(first, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    module.RAELArtifactWriter(tmp_path).append_run_jsonl(
        "loss_components.jsonl",
        _run_jsonl_row("loss_components.jsonl", microbatch_step=2),
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["microbatch_step"] for row in rows] == [1, 2]


def test_p18_complete_root_validation_holds_identity_snapshot_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "locked-complete-root"
    _write_complete_run_root(module, root)
    calls: list[str] = []
    original = module._identity_target_lock

    @contextmanager
    def recording_lock(run_root: Path, target_name: str):
        calls.append(target_name)
        with original(run_root, target_name):
            yield

    monkeypatch.setattr(module, "_identity_target_lock", recording_lock)
    module.RAELArtifactWriter(root).validate_run_root_complete()
    assert "complete-validation" in calls
