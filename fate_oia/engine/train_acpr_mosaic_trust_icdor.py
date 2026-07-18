from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex, load_bdd100k_objects
from fate_oia.datasets.mosaic_icdor_grounding import ICDORGroundingObservationBuilder
from fate_oia.datasets.mosaic_icdor_factor_supervision import build_factor_supervision
from fate_oia.datasets.mosaic_multiview import MOSAICWeakMultiView
from fate_oia.datasets.mosaic_icdor_split import (
    DEFAULT_SEED,
    ICDORTrainSplits,
    make_icdor_train_splits,
    write_icdor_split_manifest,
)
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.engine.mosaic_icdor_schedule import ICDORPhase
from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule
from fate_oia.engine.build_mosaic_edge_admission import (
    MOSAICEdgeInterventionStats,
    build_edge_admission,
)
from fate_oia.engine.build_mosaic_factor_certificate import build_and_write_factor_certificate
from fate_oia.engine.build_mosaic_visual_credibility import refresh_model_visual_credibility
from fate_oia.engine.build_mosaic_target_utility import build_target_utility
from fate_oia.engine.eval_acpr_mosaic_trust_icdor import evaluate_icdor
from fate_oia.engine.export_mosaic_trust_visual_audit import export_visual_audit
from fate_oia.engine.mosaic_icdor_audit_collectors import (
    collect_edge_intervention_audit,
    collect_factor_audit,
)
from fate_oia.engine.mosaic_target_transfer_metrics import collect_joint_target_transfer_metrics
from fate_oia.engine.mosaic_icdor_hidden_recovery_audit import (
    MISSINGNESS_MODES,
    audit_hidden_recovery_scores,
    build_hidden_mask,
)
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.losses.mosaic_icdor_action_losses import action_base_losses, action_route_losses
from fate_oia.losses.mosaic_icdor_factor_losses import (
    factor_contradiction_consistency_loss,
    factor_positive_anchor_loss,
    factor_geometry_alignment_loss,
    factor_presence_visibility_losses,
    factor_prototype_regularization,
    factor_view_consistency_loss,
)
from fate_oia.losses.mosaic_icdor_reason_losses import (
    build_synthetic_hidden_positive_mask,
    reason_observed_losses,
    selective_observation_losses,
)
from fate_oia.losses.mosaic_posterior_ranking import (
    action_cross_image_ranking_loss,
    posterior_weighted_reason_ranking_loss,
)
from fate_oia.models.acpr_mosaic_trust_icdor_model import MOSAICTrustICDORModel
from fate_oia.optim.mosaic_action_pareto_admission import MOSAICActionParetoAdmission
from fate_oia.optim.mosaic_soft_rank_queue import MOSAICAccumulationQueueBuffer, MOSAICSoftRankQueue
from fate_oia.utils.mosaic_icdor_artifacts import (
    initialize_icdor_run_artifacts,
    validate_icdor_artifact_schema,
    write_icdor_adaptive_schedule_transition,
    write_icdor_epoch_artifacts,
)
from fate_oia.utils.mosaic_config_usage import ConfigUsageTracker, resolve_icdor_config_tree


ICDOR_OWNER_GROUPS = (
    "visual_pyramid",
    "factor_adapter",
    "factor_extractor",
    "factor_prototypes",
    "action_adapter",
    "action_visual_decoder",
    "action_router_rereader",
    "reason_adapter",
    "reason_visual_decoder",
    "reason_latent_decoder",
    "reason_observed_mixer",
    "observation_model",
    "threshold_head",
)

ICDOR_REQUIRED_REMEDIATION_GATES = (
    "CANONICAL_MULTIVIEW", "REAL_FACTOR_AUDIT", "HARD_MASK_INVARIANCE",
    "PARETO_FIREWALL", "HIDDEN_RECOVERY_NO_LEAKAGE", "MATCHED_CONTROL_CCA",
    "CONFIG_COVERAGE", "QUEUE_TIMING", "ADAPTIVE_SCHEDULE", "RUNTIME_PROFILE",
    "PILOT", "STRICT_ARTIFACT_VALIDATION",
)


def resolve_icdor_policy_weights(policy: Mapping[str, float] | None = None) -> dict[str, float]:
    """Resolve route/loss policy once so scheduler decisions affect training."""
    defaults = {
        "action_visual": 1.0,
        # The old cross-image queue rank is retained only as a diagnostic;
        # v4 uses the same-image selected-vs-random matched control instead.
        "action_rank": 0.0,
        "action_shadow": 0.50,
        "reason_visual_observed": 1.0,
        "reason_observation": 0.20,
        "reason_posterior_rank": 0.0,
        "factor_presence": 1.0,
    }
    if policy is not None:
        defaults.update({str(key): float(value) for key, value in policy.items()})
    return defaults


def clip_icdor_owner_gradients(
    owner_parameters: Mapping[str, Iterable[nn.Parameter]],
    clip_limits: Mapping[str, float],
) -> dict[str, float]:
    """Clip each owner independently and return pre-clip norms."""
    norms: dict[str, float] = {}
    for owner, parameters in owner_parameters.items():
        parameters = [parameter for parameter in parameters if parameter.grad is not None]
        if not parameters:
            norms[owner] = 0.0
            continue
        norm = torch.linalg.vector_norm(torch.stack([parameter.grad.detach().norm() for parameter in parameters]))
        norms[owner] = float(norm.detach().cpu())
        limit = float(clip_limits.get(owner, 1.0))
        if limit <= 0.0:
            raise ValueError("IC-DOR owner gradient clip limits must be positive")
        scale = min(1.0, limit / float(norm.detach().cpu().clamp_min(1e-12)))
        if scale < 1.0:
            for parameter in parameters:
                parameter.grad.mul_(scale)
    return norms


def _pending_evidence_document(artifact: str, *, build_epoch: int) -> dict[str, Any]:
    """Persist an honest pre-collection state without fabricating audit evidence."""
    if artifact not in {"factor_certificate", "edge_admission"} or build_epoch < 0:
        raise ValueError("IC-DOR pending evidence document is invalid")
    return {
        "artifact": artifact,
        "status": "pending",
        "available": False,
        "source_split": None,
        "build_epoch": build_epoch,
        "reason": "scheduled_train_audit_collection_not_completed",
    }


def _edge_statistics_from_audit(payload: dict[str, Any]) -> dict[tuple[str, str, str], MOSAICEdgeInterventionStats]:
    """Translate persisted collector output without substituting missing metrics."""
    if payload.get("source_split") not in {"audit_target", "train_audit"} or not isinstance(payload.get("edge_stats"), dict):
        raise ValueError("IC-DOR edge statistics must come from an independent target audit")
    converted: dict[tuple[str, str, str], MOSAICEdgeInterventionStats] = {}
    for record in payload["edge_stats"].values():
        if record.get("available") is not True:
            continue
        metrics = record.get("metrics", {})
        lcb = record.get("bootstrap_lcb95", {})
        counts = record.get("matched_counts", {})
        required_metrics = {"cca", "isolated_edge_ap", "visual_ap"}
        required_lcb = {"signed_effect", "tet", "tes", "tes_identity", "tes_spatial"}
        if not required_metrics <= set(metrics) or not required_lcb <= set(lcb):
            raise ValueError("IC-DOR edge audit is missing real admission metrics")
        valid_samples = min(int(counts.get(name, 0)) for name in ("factor_on", "factor_off", "equal_mass_random"))
        key = (str(record["direction"]), str(record["factor"]), str(record["action"]))
        converted[key] = MOSAICEdgeInterventionStats(
            valid_samples=valid_samples,
            signed_effect_lcb95=float(lcb["signed_effect"]),
            tet_lcb95=float(lcb["tet"]),
            tes_lcb95=float(lcb["tes"]),
            tes_identity_lcb95=float(lcb["tes_identity"]),
            tes_spatial_lcb95=float(lcb["tes_spatial"]),
            cca=float(metrics["cca"]),
            isolated_edge_ap=float(metrics["isolated_edge_ap"]),
            visual_ap=float(metrics["visual_ap"]),
        )
    return converted


def _factor_audit_rows(payload: dict[str, Any], *, epoch: int) -> list[dict[str, Any]]:
    if payload.get("source_split") != "audit_visual" or not isinstance(payload.get("factor_stats"), dict):
        raise ValueError("CREDO per-epoch factor diagnostics must come from audit_visual")
    rows = []
    for factor_name, stats in payload["factor_stats"].items():
        if not isinstance(stats, dict) or not {"counts", "scores", "prototype", "bootstrap_lcb95"} <= set(stats):
            raise ValueError(f"IC-DOR factor diagnostic is incomplete for {factor_name}")
        rows.append({"epoch": epoch, "source_split": "audit_visual", "factor_name": factor_name, **stats})
    if not rows:
        raise ValueError("IC-DOR factor audit produced no factor rows")
    return rows


def _restore_phase_trainability(model: nn.Module, phase: ICDORPhase) -> None:
    """Undo calibration-only freezing, then apply the canonical phase firewall."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("dino."))
    if phase.freeze_factor_branch:
        apply_icdor_factor_branch_freeze(model)


def apply_icdor_consolidation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """Freeze propensity and reduce decoder/router learning rates exactly once."""
    for parameter in model.observation_model.parameters():
        parameter.requires_grad_(False)
    reduced = {
        "action_visual_decoder", "action_router_rereader", "reason_visual_decoder",
        "reason_latent_decoder", "reason_observed_mixer",
    }
    for index, group in enumerate(optimizer.param_groups):
        if group.get("name") in reduced:
            scheduler.base_lrs[index] *= 0.2
            group["lr"] *= 0.2


def apply_icdor_route_gate_schedule(model: nn.Module, phase: ICDORPhase) -> float:
    """Keep route capacity coupled to adaptive state, never a wall-clock epoch."""
    cap = {"off": 0.02, "shadow": 0.05, "admitted": 0.05}[phase.route_mode]
    model.set_route_gate_cap(cap)
    return cap


def _adaptive_phase(schedule: ICDORAdaptiveSchedule) -> ICDORPhase:
    return schedule.phase()


def _finite_scalar_rows(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _factor_audit_integrity_metrics(factor_audit: dict[str, Any]) -> dict[str, bool]:
    """Turn explicit collector evidence into fail-closed readiness flags."""
    integrity = factor_audit.get("audit_integrity")
    stats = factor_audit.get("factor_stats")
    if not isinstance(integrity, dict) or not isinstance(stats, dict) or not stats:
        return {
            "factor_audit_complete": False,
            "factor_audit_exception": True,
            "unknown_abstained": False,
        }
    completed = integrity.get("collector_completed") is True
    exception_free = integrity.get("exception") is None
    unknown_total = sum(
        int(row.get("counts", {}).get("unknown", 0))
        for row in stats.values() if isinstance(row, dict)
    )
    unknown_proof = (
        integrity.get("unknown_policy") == "excluded_from_binary_metrics"
        and int(integrity.get("unknown_rows_total", -1)) == unknown_total
        and int(integrity.get("unknown_rows_in_metric_total", -1)) == 0
    )
    return {
        "factor_audit_complete": bool(
            factor_audit.get("source_split") in {"audit_visual", "train_audit"} and completed and exception_free
        ),
        "factor_audit_exception": not bool(completed and exception_free),
        "unknown_abstained": bool(completed and exception_free and unknown_proof),
    }


def _adaptive_readiness(
    *,
    epoch_train: dict[str, list[dict[str, Any]]],
    factor_audit: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    pareto_rows: list[dict[str, Any]],
    certificate_sha256: str | None,
    edge_admission_sha256: str | None,
    certificate_snapshot: dict[str, Any],
    previous_certificate_tiers: dict[str, str] | None,
    ontology: dict[str, Any],
    branch_readiness: dict[str, Any],
    edge_document: dict[str, Any],
    previous_best_train_audit_joint: float | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive scheduler inputs from explicitly separated audit evidence.

    The historical schedule key remains ``train_audit`` for checkpoint
    compatibility, but it is an aggregate document only: visual credibility
    must come from ``audit_visual`` and label-dependent target utility must
    come from ``audit_target``.
    """
    if factor_audit.get("source_split") != "audit_visual":
        raise ValueError("CREDO readiness requires audit_visual factor evidence")
    if branch_readiness.get("source_split") != "audit_target":
        raise ValueError("CREDO readiness requires audit_target branch evidence")
    factor_stats = factor_audit.get("factor_stats", {})
    diagnostic_values = [
        value
        for stats in factor_stats.values()
        if isinstance(stats, dict)
        for section in (stats.get("scores", {}), stats.get("bootstrap_lcb95", {}))
        if isinstance(section, dict)
        for value in section.values()
        if value is not None
    ]
    pareto = pareto_rows[0] if pareto_rows and pareto_rows[0].get("available") is True else {}
    train_core = {
        "source_split": "train_core",
        "finite": _finite_scalar_rows(epoch_train["loss_rows"]) and _finite_scalar_rows(epoch_train["runtime_rows"]),
    }
    train_calib = {
        "source_split": "train_calib",
        "finite": _finite_scalar_rows(calibration_rows),
    }
    certificate_metrics = _certificate_readiness_metrics(
        certificate=certificate_snapshot,
        previous_tiers=previous_certificate_tiers,
        factor_stats=factor_stats,
        ontology=ontology,
    )
    route_metrics = _edge_and_branch_readiness_metrics(
        edge_document=edge_document,
        branch_metrics=branch_readiness,
        previous_best_train_audit_joint=previous_best_train_audit_joint,
    )
    factor_integrity = _factor_audit_integrity_metrics(factor_audit)
    train_audit = {
        "source_split": "train_audit",
        "audit_visual_source_split": "audit_visual",
        "audit_target_source_split": "audit_target",
        **factor_integrity,
        **certificate_metrics,
        "diagnostics_finite": bool(diagnostic_values) and all(math.isfinite(float(value)) for value in diagnostic_values),
        **route_metrics,
        "pareto_violation_rate": float(pareto.get("pareto_violation_rate", 1.0)),
    }
    return train_core, train_audit, train_calib


def _pilot_semantic_validation(
    schedule_state: dict[str, Any],
    certificate: dict[str, Any],
    edge_document: dict[str, Any],
) -> dict[str, Any]:
    """Validate pilot learning access separately from final deployment admission."""
    errors: list[str] = []
    history = schedule_state.get("history")
    history = history if isinstance(history, list) else []
    foundation_transition = next((
        row for row in history
        if row.get("state_before") == "FOUNDATION"
        and row.get("state_after") == "DUAL_REASON_SHADOW"
    ), None)
    shadow_rows = [
        row for row in history
        if row.get("state_after") in {"DUAL_REASON_SHADOW", "SAFE_JOINT", "CONSOLIDATION"}
    ]

    def _split_integrity(row: dict[str, Any]) -> bool:
        readiness = row.get("readiness", {})
        train_core = readiness.get("train_core", {})
        train_calib = readiness.get("train_calib", {})
        return (
            train_core.get("source_split") == "train_core"
            and train_core.get("finite") is True
            and train_calib.get("source_split") == "train_calib"
            and train_calib.get("finite") is True
        )
    if foundation_transition is None:
        errors.append("missing_foundation_to_shadow_transition")
    elif not _split_integrity(foundation_transition):
        errors.append("foundation_transition_lacks_split_integrity")
    if schedule_state.get("failed_closed") is True:
        errors.append("adaptive_schedule_failed_closed")
    learning_row = next((
        row for row in reversed(shadow_rows)
        if isinstance(row.get("readiness", {}).get("train_audit"), dict)
        and row["readiness"]["train_audit"].get("source_split") == "train_audit"
    ), None)
    if learning_row is None or not _split_integrity(learning_row):
        errors.append("missing_or_invalid_shadow_learning_access")
    else:
        readiness = learning_row.get("readiness", {}).get("train_audit", {})
        action_delta = readiness.get("action_shadow_ap_delta", [])
        action_nonregression = sum(float(value) >= -0.002 for value in action_delta) >= 2
        route_strength = float(readiness.get("route_strength_ratio", -1.0))
        if not (
            readiness.get("continuous_credibility_available") is True
            and int(readiness.get("observable_cV_gt_030", 0)) >= 6
            and readiness.get("diagnostics_finite") is True
            and 0.005 <= route_strength <= 0.10
            and readiness.get("factor_shuffle_degrades_reason") is True
            and action_nonregression
            and readiness.get("disallowed_route_invariance") is True
        ):
            errors.append("shadow_learning_access_metrics_are_incomplete")

    edge_entries = edge_document.get("entries")
    edge_entries = edge_entries if isinstance(edge_entries, dict) else {}
    accepted_actions = {
        str(row.get("target")) for row in edge_entries.values()
        if isinstance(row, dict) and row.get("accepted") is True and row.get("target") is not None
    }
    certificate_entries = certificate.get("entries")
    certificate_entries = certificate_entries if isinstance(certificate_entries, dict) else {}
    deployment_admission_ready = bool(accepted_actions) and any(
        isinstance(row, dict) and row.get("tier") == "certified" for row in certificate_entries.values()
    )
    return {
        "pass": not errors,
        "errors": errors,
        "accepted_action_count": len(accepted_actions),
        "deployment_admission_ready": deployment_admission_ready,
        "history_length": len(history),
    }


def _write_provisional_certificate_this_epoch(
    *, write_provisional: bool, freeze_certificate: bool
) -> bool:
    """Legacy helper retained for config compatibility; CREDO never calls it in pilot."""
    if write_provisional and freeze_certificate:
        raise ValueError("IC-DOR certificate policy cannot rebuild and freeze simultaneously")
    return bool(write_provisional and not freeze_certificate)


def _certificate_tiers(certificate: dict[str, Any]) -> dict[str, str]:
    entries = certificate.get("entries")
    if not isinstance(entries, dict):
        return {}
    result: dict[str, str] = {}
    for name, row in entries.items():
        if not isinstance(row, dict) or row.get("tier") not in {"certified", "reason_only", "abstained"}:
            raise ValueError(f"IC-DOR certificate has invalid tier for {name}")
        result[str(name)] = str(row["tier"])
    return result


def _certificate_readiness_metrics(
    *,
    certificate: dict[str, Any],
    previous_tiers: dict[str, str] | None,
    factor_stats: dict[str, Any],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    """Derive FOUNDATION readiness from certificate contents, never file existence."""
    tiers = _certificate_tiers(certificate)
    certified = {name for name, tier in tiers.items() if tier == "certified"}
    reason_eligible = {name for name, tier in tiers.items() if tier in {"certified", "reason_only"}}
    route_group_counts: list[int] = []
    for action_name in ontology.get("action_names", []):
        routes = ontology.get("action_routes", {}).get(action_name, {})
        group_count = 0
        for direction in ("support", "veto"):
            edges = routes.get(direction, []) if isinstance(routes, dict) else []
            if any(isinstance(edge, dict) and str(edge.get("factor")) in certified for edge in edges):
                group_count += 1
        route_group_counts.append(group_count)
    reachable_reasons = 0
    for routes in ontology.get("reason_routes", {}).values():
        if not isinstance(routes, dict):
            continue
        factors = {
            str(name)
            for key in ("direct_factors", "latent_factors", "contradiction_factors")
            for name in routes.get(key, [])
        }
        if factors & reason_eligible:
            reachable_reasons += 1
    if previous_tiers is None:
        tier_jaccard = 0.0
    else:
        current_pairs = {(name, tier) for name, tier in tiers.items()}
        previous_pairs = {(str(name), str(tier)) for name, tier in previous_tiers.items()}
        union = current_pairs | previous_pairs
        tier_jaccard = float(len(current_pairs & previous_pairs) / len(union)) if union else 0.0
    saturated = 0
    for name in tiers:
        stats = factor_stats.get(name, {})
        scores = stats.get("scores", {}) if isinstance(stats, dict) else {}
        prototype = stats.get("prototype", {}) if isinstance(stats, dict) else {}
        presence_variance = scores.get("presence_variance")
        visibility_variance = scores.get("visibility_variance")
        dominant_rate = prototype.get("dominant_rate")
        invalid_or_saturated = (
            not isinstance(presence_variance, (int, float))
            or not isinstance(visibility_variance, (int, float))
            or not isinstance(dominant_rate, (int, float))
            or float(presence_variance) <= 0.0
            or float(visibility_variance) <= 0.0
            or float(dominant_rate) >= 0.85
        )
        saturated += int(invalid_or_saturated)
    return {
        "certified_route_group_count_per_action": route_group_counts,
        "reachable_reason_count": reachable_reasons,
        "certificate_tier_jaccard": tier_jaccard,
        "saturation_fraction": float(saturated / len(tiers)) if tiers else 1.0,
    }


def _edge_and_branch_readiness_metrics(
    *,
    edge_document: dict[str, Any],
    branch_metrics: dict[str, Any],
    previous_best_train_audit_joint: float | None,
) -> dict[str, Any]:
    """Use train-audit branch measurements and every accepted edge's real LCBs."""
    action_order = ("forward", "stop", "left", "right")
    counts = {name: 0 for name in action_order}
    accepted_metrics: list[dict[str, Any]] = []
    entries = edge_document.get("entries", {})
    if isinstance(entries, dict):
        for row in entries.values():
            if not isinstance(row, dict) or row.get("accepted") is not True:
                continue
            target = str(row.get("target"))
            metrics = row.get("metrics")
            if target not in counts or not isinstance(metrics, dict):
                raise ValueError("IC-DOR accepted edge is missing a valid target or metrics")
            required = (
                "tet_lcb95", "tes_lcb95", "tes_identity_lcb95",
                "tes_spatial_lcb95", "cca",
            )
            if any(not isinstance(metrics.get(key), (int, float)) for key in required):
                raise ValueError("IC-DOR accepted edge is missing real intervention LCBs")
            counts[target] += 1
            accepted_metrics.append(metrics)
    joint = branch_metrics.get("train_audit_joint")
    improved = (
        isinstance(joint, (int, float))
        and (
            previous_best_train_audit_joint is None
            or float(joint) > float(previous_best_train_audit_joint) + 1e-8
        )
    )
    def minimum(key: str) -> float:
        return min((float(row[key]) for row in accepted_metrics), default=0.0)
    defaults = {
        "action_shadow_ap_delta": [],
        "action_route_ap_delta": [],
        "exp_map_delta_vs_visual": -1.0,
        "factor_shuffle_degrades_reason": False,
        "hidden_recovery_margin": -1.0,
        "route_strength_ratio": 0.0,
        "disallowed_route_invariance": False,
        "exp_map_delta_vs_entry": -1.0,
    }
    result = {key: branch_metrics.get(key, value) for key, value in defaults.items()}
    result.update({
        "true_edge_count_per_action": [counts[name] for name in action_order],
        "tet_lcb95": minimum("tet_lcb95"),
        "tes_lcb95": minimum("tes_lcb95"),
        "tes_identity_lcb95": minimum("tes_identity_lcb95"),
        "tes_spatial_lcb95": minimum("tes_spatial_lcb95"),
        "cca": minimum("cca"),
        "train_audit_improved": improved,
    })
    return result


def apply_icdor_factor_branch_freeze(model: nn.Module) -> None:
    """Freeze the complete factor measurement path after its audit certificate is frozen."""
    prefixes = ("factor_visual_pyramid.", "factor_adapter.", "factor_extractor.")
    matched = 0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(False)
            matched += 1
    if matched == 0:
        raise ValueError("IC-DOR factor branch freeze found no factor parameters")


def _require(actual: dict[str, Any], expected: dict[str, Any], prefix: str = "") -> None:
    for key, expected_value in expected.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in actual:
            raise ValueError(f"IC-DOR config missing {name}")
        value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(value, dict):
                raise ValueError(f"IC-DOR config {name} must be a mapping")
            _require(value, expected_value, name)
        elif value != expected_value:
            raise ValueError(f"IC-DOR config drift at {name}: expected {expected_value!r}, got {value!r}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the formal IC-DOR contract and reject protocol drift before any run."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    expected = {
        "experiment": {
            "name": "acpr_mosaic_trust_v3_icdor",
            "direct_image": True,
            "initialization": "public_dino_vits8_teacher_only",
            "eval_splits": ["test"],
            "best_selection_split": "test",
            "best_selection_metric": "deploy_fixed_joint",
            "no_metric_early_stop": True,
        },
        "data": {
            "image_height": 360,
            "image_width": 640,
            "patch_size": 8,
            "action_dim": 4,
            "reason_dim": 21,
            "train_core_fraction": 0.80,
            "train_audit_fraction": 0.10,
            "train_calib_fraction": 0.10,
            "split_seed": DEFAULT_SEED,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },
        "backbone": {
            "arch": "vit_small",
            "patch_size": 8,
            "selected_layers": [3, 7, 11],
            "checkpoint_key": "teacher",
            "freeze_backbone": True,
            "no_grad_backbone": True,
            "feature_cache": False,
            "token_compression": "none",
        },
        "model": {
            "formal_class": "MOSAICTrustICDORModel",
            "dim": 384,
            "separate_visual_pyramids": True,
            "action_set_final": False,
        },
        "credibility": {
            "independent_of_reason_labels": True,
            "observable_cV_min_for_admission": 0.80,
            "bootstrap_replicates": 1000,
            "ema_decay": 0.90,
            "image_only_cap": 0.10,
            "unknown_cap": 0.0,
            "no_reliable_negative_cap": 0.25,
        },
        "fine_transport": {
            "enabled": True,
            "point_eta": 0.70,
            "curve_eta": 0.80,
            "region_eta": 0.60,
            "local_reread_offset_max": 0.08,
            "fine_off_diagnostic": True,
            "coarse_off_diagnostic": True,
        },
        "training": {"epochs": 12, "no_metric_early_stop": True},
        "calibration": {
            "enabled": True,
            "train_calib_only": True,
            "representation_frozen": True,
            "deploy_equation": "raw_minus_theta",
            "test_oracle_diagnostic_only": True,
        },
        "evaluation": {
            "eval_splits": ["test"],
            "best_selection_split": "test",
            "best_selection_metric": "deploy_fixed_joint",
        },
        "runtime": {
            "foreground_only": True,
            "no_feature_cache": True,
            "require_no_token_compression": True,
        },
    }
    _require(config, expected)
    if config.get("model", {}).get("use_action_state_delta") is True:
        raise ValueError("IC-DOR formal action must not use a state delta")
    if config.get("model", {}).get("action_set_final") is not False:
        raise ValueError("IC-DOR action-set head is auxiliary only")
    return config


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([row["image"] for row in batch]),
        "action": torch.stack([row["action"] for row in batch]),
        "reason": torch.stack([row["reason"] for row in batch]),
        "file_name": [row["file_name"] for row in batch],
        "image_path": [row["image_path"] for row in batch],
        "split": [row["split"] for row in batch],
    }


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    config: dict[str, Any],
    generator: torch.Generator | None = None,
) -> DataLoader:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("IC-DOR loader requires positive batch_size and non-negative num_workers")
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(config["data"]["pin_memory"]),
        "collate_fn": _collate,
        "drop_last": bool(shuffle),
        "generator": generator,
    }
    if num_workers:
        kwargs.update(
            persistent_workers=bool(config["data"]["persistent_workers"]),
            prefetch_factor=int(config["data"]["prefetch_factor"]),
            timeout=900,
        )
    return DataLoader(dataset, **kwargs)


def _subset_indices(indices: Iterable[int], limit: int | None) -> list[int]:
    ordered = list(indices)
    if limit is None or limit >= len(ordered):
        return ordered
    if type(limit) is not int or limit <= 0:
        raise ValueError("IC-DOR subset limit must be a positive integer")
    return ordered[:limit]


def _factor_aware_audit_subset(
    dataset: BDDOIAMultiTaskDataset,
    indices: Iterable[int],
    limit: int | None,
    *,
    grounding_index: BDD100KGroundingIndex,
    seed: int,
) -> list[int]:
    """Geometry-only, deterministic pilot subset balancing observable factor families."""
    candidates = list(indices)
    if limit is None or limit >= len(candidates):
        return candidates
    if type(limit) is not int or limit <= 0:
        raise ValueError("IC-DOR factor-aware audit limit must be a positive integer")

    object_tokens = ("car", "truck", "bus", "person", "pedestrian", "rider", "bike", "motor")
    traffic_tokens = ("traffic light", "traffic sign", "sign", "light")
    tags: dict[int, tuple[bool, bool, bool]] = {}
    for index in candidates:
        sample = dataset.samples[index]
        file_name = sample["file_name"] if isinstance(sample, dict) else sample.file_name
        paths = grounding_index.lookup(str(file_name))
        categories: list[str] = []
        if paths.label_json:
            for item in load_bdd100k_objects(paths.label_json):
                category = item.get("category") or item.get("label") or item.get("name")
                if isinstance(category, str):
                    categories.append(category.lower())
        joined = " ".join(categories)
        tags[index] = (
            any(token in joined for token in object_tokens),
            any(token in joined for token in traffic_tokens),
            paths.drivable_map is not None,
        )

    def rank(index: int) -> tuple[int, int]:
        sample = dataset.samples[index]
        file_name = sample["file_name"] if isinstance(sample, dict) else sample.file_name
        digest = hashlib.sha256(f"{seed}:{file_name}".encode("utf-8")).digest()
        return int.from_bytes(digest, "big", signed=False), index

    availability = [sum(values[group] for values in tags.values()) for group in range(3)]
    targets = [max(1, round(total * limit / len(candidates))) if total else 0 for total in availability]
    selected: set[int] = set()
    selected_counts = [0, 0, 0]
    while len(selected) < limit:
        deficits = [max(targets[group] - selected_counts[group], 0) for group in range(3)]
        group = max(range(3), key=lambda item: (deficits[item], -item))
        if deficits[group] <= 0:
            break
        pool = sorted((index for index in candidates if tags[index][group] and index not in selected), key=rank)
        if not pool:
            targets[group] = selected_counts[group]
            continue
        chosen = pool[0]
        selected.add(chosen)
        for item, value in enumerate(tags[chosen]):
            selected_counts[item] += int(value)
    for index in sorted(candidates, key=rank):
        if len(selected) >= limit:
            break
        selected.add(index)
    if len(selected) != limit:
        raise RuntimeError("IC-DOR factor-aware audit sampler returned the wrong subset size")
    return sorted(selected)


def build_icdor_loaders(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    max_train_samples: int | None = None,
    max_audit_samples: int | None = None,
    max_calib_samples: int | None = None,
    max_test_samples: int | None = None,
    visual_grounding_index: BDD100KGroundingIndex | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    """Build core, disjoint visual/target audit, calib, and test loaders."""
    data = config["data"]
    transform = AspectRatioLetterboxTransform(
        data["image_height"], data["image_width"], patch_size=data["patch_size"], normalize=True
    )
    train = BDDOIAMultiTaskDataset(
        data["data_root"], data["raw_root"], split="train", action_dim=4, reason_dim=21,
        load_image=True, transform=transform,
    )
    test = BDDOIAMultiTaskDataset(
        data["data_root"], data["raw_root"], split="test", action_dim=4, reason_dim=21,
        load_image=True, transform=transform,
    )
    split: ICDORTrainSplits = make_icdor_train_splits(train, seed=int(data["split_seed"]))
    split_dir = Path(output_dir) / "split"
    write_icdor_split_manifest(split, split_dir / "icdor_split_manifest.json")
    core = _subset_indices(split.train_core_indices, max_train_samples)
    # cV uses only image/BDD100K geometry observations. Never use reason
    # labels to select this pilot population.
    # Reuse the caller's immutable BDD100K index when subsequent epoch-end
    # audits need the same data. Rebuilding it here used to add a second full
    # JSON scan before the first DINO forward.
    visual_grounding = visual_grounding_index or BDD100KGroundingIndex(data["bdd100k_root"])
    audit_visual = _factor_aware_audit_subset(
        train, split.audit_visual_indices, max_audit_samples,
        grounding_index=visual_grounding, seed=int(data["split_seed"]),
    )
    audit_target = _subset_indices(split.audit_target_indices, max_audit_samples)
    calib = _subset_indices(split.train_calib_indices, max_calib_samples)
    test_indices = _subset_indices(range(len(test)), max_test_samples)
    if (
        not set(core).isdisjoint(audit_visual)
        or not set(core).isdisjoint(audit_target)
        or not set(core).isdisjoint(calib)
        or not set(audit_visual).isdisjoint(audit_target)
        or not set(audit_visual).isdisjoint(calib)
        or not set(audit_target).isdisjoint(calib)
    ):
        raise RuntimeError("IC-DOR train_core/audit_visual/audit_target/train_calib must remain disjoint")
    seed = int(data["split_seed"])
    stats = {
        "split_sha256": split.split_sha256,
        "split_seed": seed,
        "train_core_count": len(core),
        "train_audit_count": len(audit_visual) + len(audit_target),
        "audit_visual_count": len(audit_visual),
        "audit_target_count": len(audit_target),
        "train_calib_count": len(calib),
        "test_count": len(test_indices),
        "train_audit_positive_counts": list(split.audit_positive_counts),
        "audit_visual_positive_counts": list(split.audit_visual_positive_counts),
        "audit_target_positive_counts": list(split.audit_target_positive_counts),
        "train_calib_positive_counts": list(split.calib_positive_counts),
    }
    return (
        _loader(Subset(train, core), batch_size=batch_size, shuffle=True, num_workers=num_workers, config=config,
                generator=torch.Generator().manual_seed(seed)),
        _loader(Subset(train, audit_visual), batch_size=batch_size, shuffle=False, num_workers=num_workers, config=config),
        _loader(Subset(train, audit_target), batch_size=batch_size, shuffle=False, num_workers=num_workers, config=config),
        _loader(Subset(train, calib), batch_size=batch_size, shuffle=False, num_workers=num_workers, config=config),
        _loader(Subset(test, test_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, config=config),
        stats,
    )


def _owner_for_name(name: str) -> tuple[str, tuple[str, ...]] | None:
    mapping: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("factor_visual_pyramid.", "visual_pyramid", ("factor",)),
        ("action_visual_pyramid.", "visual_pyramid", ("action_base",)),
        ("reason_visual_pyramid.", "visual_pyramid", ("reason_observed",)),
        ("factor_adapter.", "factor_adapter", ("factor",)),
        ("factor_extractor.prototype_bank.", "factor_prototypes", ("factor",)),
        ("continuous_credibility.", "factor_extractor", ("factor",)),
        ("factor_extractor.", "factor_extractor", ("factor",)),
        ("action_adapter.", "action_adapter", ("action_base",)),
        ("action_visual_decoder.", "action_visual_decoder", ("action_base",)),
        ("action_router.", "action_router_rereader", ("action_route",)),
        ("action_rereader.", "action_router_rereader", ("action_route",)),
        ("reason_adapter.", "reason_adapter", ("reason_observed",)),
        ("reason_visual_decoder.", "reason_visual_decoder", ("reason_observed",)),
        ("reason_latent_decoder.", "reason_latent_decoder", ("reason_latent",)),
        ("reason_observed_mixer.", "reason_observed_mixer", ("reason_observed",)),
        ("observation_model.", "observation_model", ("reason_observation",)),
        ("threshold_head.", "threshold_head", ("calibration",)),
    )
    matches = [(prefix, owner, losses) for prefix, owner, losses in mapping if name.startswith(prefix)]
    if not matches:
        return None
    longest = max(len(prefix) for prefix, _, _ in matches)
    winners = [(owner, losses) for prefix, owner, losses in matches if len(prefix) == longest]
    if len(winners) != 1:
        raise ValueError(f"ambiguous IC-DOR parameter ownership for {name}")
    return winners[0]


def build_icdor_parameter_ownership(
    model: nn.Module,
    *,
    output_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[nn.Parameter]]]:
    """Assign every trainable tensor to one optimizer group, never two."""
    groups: dict[str, list[nn.Parameter]] = {group: [] for group in ICDOR_OWNER_GROUPS}
    ownership: list[dict[str, Any]] = []
    seen_parameters: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        owner = _owner_for_name(name)
        if owner is None:
            raise ValueError(f"unassigned IC-DOR trainable parameter: {name}")
        owner_group, allowed_losses = owner
        if id(parameter) in seen_parameters:
            raise ValueError(f"duplicate IC-DOR trainable parameter: {name}")
        seen_parameters.add(id(parameter))
        groups[owner_group].append(parameter)
        ownership.append({
            "full_name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "owner_group": owner_group,
            "allowed_losses": list(allowed_losses),
            "requires_grad_by_phase": {"foundation": True, "formal": True, "calibration": owner_group == "threshold_head"},
        })
    missing = [group for group, parameters in groups.items() if not parameters]
    if missing:
        raise ValueError(f"IC-DOR optimizer ownership group has no parameters: {missing}")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"parameters": ownership}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ownership, groups


def build_icdor_optimizer(model: nn.Module, config: dict[str, Any]) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    ownership, groups = build_icdor_parameter_ownership(model)
    lr_config = config["optimizer"]["lr"]
    param_groups: list[dict[str, Any]] = []
    for group_name in ICDOR_OWNER_GROUPS:
        parameters = groups[group_name]
        param_groups.append({"params": parameters, "lr": float(lr_config[group_name]), "name": group_name})
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(config["optimizer"]["weight_decay"]),
            fused=bool(config["optimizer"].get("fused_when_supported", False)) and torch.cuda.is_available(),
        )
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(param_groups, weight_decay=float(config["optimizer"]["weight_decay"]))
    return optimizer, ownership


def build_icdor_model(config: dict[str, Any], *, use_mock_dino: bool = False, mock_dim: int = 384) -> MOSAICTrustICDORModel:
    model_config = config["model"]
    adapter = model_config["adapter"]
    typed = model_config["typed_attention"]
    measurement = model_config["factor_measurement"]
    route = model_config["action_route"]
    reason = model_config["reason"]
    observation = config["selective_observation"]
    backbone = config["backbone"]
    credibility = config["credibility"]
    fine_transport = config["fine_transport"]
    return MOSAICTrustICDORModel(
        config_root=Path("configs"),
        backbone_arch=str(backbone["arch"]),
        backbone_patch_size=int(backbone["patch_size"]),
        selected_layers=tuple(int(value) for value in backbone["selected_layers"]),
        checkpoint_key=str(backbone["checkpoint_key"]),
        pretrained_weights=str(backbone["pretrained_weights"]),
        use_mock_dino=use_mock_dino,
        mock_dim=mock_dim,
        decoder_layers=int(model_config["decoder_layers"]),
        self_attention_heads=int(model_config["self_attention_heads"]),
        anchors_per_factor=int(typed["anchors_per_factor"]),
        typed_attention_heads=int(typed["heads"]),
        point_samples=int(typed["point_samples"]),
        curve_samples=int(typed["curve_samples"]),
        region_samples=int(typed["region_samples"]),
        adapter_rank=min(int(adapter["rank"]), mock_dim) if use_mock_dino else int(adapter["rank"]),
        adapter_dropout=float(adapter["dropout"]),
        adapter_rezero_init=float(adapter["rezero_init"]),
        adapter_rezero_max=float(adapter["rezero_max"]),
        spatial_prior_scale_init=float(measurement["spatial_prior_scale_init"]),
        spatial_prior_scale_max=float(measurement["spatial_prior_scale_max"]),
        spatial_prior_dropout=float(measurement["spatial_prior_dropout"]),
        content_temperature_init=float(measurement["content_temperature_init"]),
        gate_init=float(route["gate_init"]),
        gate_max=float(route["gate_max"]),
        pi_min=float(observation["pi_min"]),
          pi_max=float(observation["pi_max"]),
          observed_mix_init=float(reason["observed_mix_init"]),
          credibility_independent_of_reason_labels=bool(credibility["independent_of_reason_labels"]),
          action_credibility_min_for_admission=float(credibility["observable_cV_min_for_admission"]),
          credibility_ema_decay=float(credibility["ema_decay"]),
          credibility_image_only_cap=float(credibility["image_only_cap"]),
          credibility_unknown_cap=float(credibility["unknown_cap"]),
          credibility_no_reliable_negative_cap=float(credibility["no_reliable_negative_cap"]),
          fine_transport_enabled=bool(fine_transport["enabled"]),
          fine_eta_by_type={
              "point": float(fine_transport["point_eta"]),
              "object": float(fine_transport["point_eta"]),
              "curve": float(fine_transport["curve_eta"]),
              "region": float(fine_transport["region_eta"]),
          },
          local_reread_offset_max=float(fine_transport["local_reread_offset_max"]),
          fine_off_diagnostic=bool(fine_transport["fine_off_diagnostic"]),
          coarse_off_diagnostic=bool(fine_transport["coarse_off_diagnostic"]),
      )


def build_icdor_contradiction_mask(model: MOSAICTrustICDORModel) -> torch.Tensor:
    factors = model.ontology["factors"]
    index = model.ontology["factor_index"]
    mask = torch.zeros(len(factors), len(factors), dtype=torch.bool)
    for factor in factors:
        source = index[factor["name"]]
        for target_name in factor.get("contradicts", []):
            target = index[target_name]
            mask[source, target] = True
            mask[target, source] = True
    return mask


def build_icdor_multiview_batch(
    images: torch.Tensor,
    transform: MOSAICWeakMultiView,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    first: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    first_metadata: list[dict[str, Any]] = []
    second_metadata: list[dict[str, Any]] = []
    for image in images:
        result = transform(image)
        first.append(result["images"][0])
        second.append(result["images"][1])
        first_metadata.append(result["metadata"][0])
        second_metadata.append(result["metadata"][1])
    return torch.stack(first), torch.stack(second), first_metadata, second_metadata


def restore_factor_view(
    values: torch.Tensor,
    metadata: list[dict[str, Any]],
    transform: MOSAICWeakMultiView,
    *,
    masks: bool,
) -> torch.Tensor:
    restored = []
    for row, item in enumerate(metadata):
        value = values[row]
        restored.append(
            transform.invert_factor_masks(value, item, factor_dim=0)
            if masks
            else transform.invert_factor_values(value, item, factor_dim=0)
        )
    return torch.stack(restored)


def compute_icdor_training_losses(
    model: MOSAICTrustICDORModel,
    output: dict[str, Any],
    second_output_restored: dict[str, torch.Tensor],
    batch: dict[str, Any],
    observations: dict[str, torch.Tensor],
    phase: ICDORPhase,
    pareto: MOSAICActionParetoAdmission,
    action_queue: MOSAICSoftRankQueue,
    reason_queue: MOSAICSoftRankQueue,
    *,
    hidden_mask: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    action_targets = batch["action"]
    reason_targets = batch["reason"]
    sample_ids = batch["file_name"]
    losses: dict[str, torch.Tensor] = {}
    loss_config = (config or {}).get("loss", {})
    policy = resolve_icdor_policy_weights(
        loss_config.get("policy") if isinstance(loss_config.get("policy"), Mapping) else None
    )
    action_config = loss_config.get("action", {})
    route_config = loss_config.get("action_route", {})
    factor_config = loss_config.get("factor", {})
    reason_config = loss_config.get("reason", {})
    base = action_base_losses(
        output["action_visual_logits"], action_targets,
        gamma_pos=float(action_config.get("gamma_pos", 0.0)),
        gamma_neg=float(action_config.get("gamma_neg", 4.0)),
        clip=float(action_config.get("clip", 0.05)),
        base_asl_weight=float(action_config.get("base_asl_weight", 1.0)),
        # The queue-backed cross-image rank below is the sole rank term.
        rank_weight=0.0,
        cardinality_weight=float(action_config.get("cardinality_weight", 0.02)),
    )
    action_rank, action_rank_stats = action_cross_image_ranking_loss(
        output["action_visual_logits"], action_targets, sample_ids, action_queue
    )
    losses.update(base)
    losses["loss_action_cross_sample_rank"] = action_rank
    action_total = policy["action_visual"] * base["loss_action_base_total"] + policy["action_rank"] * action_rank
    if phase.route_mode != "off":
        pareto_penalty = pareto.route_penalty(
            output["action_visual_logits"], output["action_shadow_logits"], action_targets
        ) if phase.enable_pareto else output["action_shadow_logits"].sum() * 0.0
        route = action_route_losses(
            output["action_visual_logits"],
            output["action_support_logits"],
            output["action_veto_logits"],
            action_targets,
            support_dustbin=output["support_dustbin"],
            veto_dustbin=output["veto_dustbin"],
            pareto_penalty=pareto_penalty,
            matched_random_logits=output["action_matched_random_logits"],
            route_strength_target=float(config["model"]["action_route"]["route_strength_target"]) if config else 0.05,
            shadow_asl_weight=float(route_config.get("shadow_asl_weight", policy["action_shadow"])),
            pareto_weight=float(route_config.get("pareto_weight", 1.0)),
            sparsity_weight=float(route_config.get("sparsity_weight", 0.02)),
            dustbin_weight=float(route_config.get("dustbin_weight", 0.01)),
            strength_weight=float(route_config.get("strength_weight", 0.02)),
            intervention_weight=float(route_config.get("intervention_weight", 0.05)),
        )
        losses.update(route)
        action_total = action_total + route["loss_action_route_total"]
    factor_total = output["factor_presence_logits"].sum() * 0.0
    if phase.enable_factor_losses:
        split_values = batch.get("split", ["train_core"] * action_targets.shape[0])
        if not isinstance(split_values, list) or not split_values or len(set(split_values)) != 1:
            raise ValueError("IC-DOR factor supervision requires a homogeneous declared split")
        supervision = build_factor_supervision(
            observations,
            reason_targets,
            model.ontology["factors"],
            split=str(split_values[0]),
            # Reasons are target-side observations, not visual factor labels.
            # Legacy anchor behavior stays opt-in for reproducibility only.
            allow_reason_anchors=bool(factor_config.get("allow_reason_anchors", False)),
        )
        factor = factor_presence_visibility_losses(
            output["factor_presence_logits"], output["factor_visibility_logits"],
            observations["presence_target"], observations["visibility_target"],
            observations["presence_known_mask"], observations["visibility_known_mask"],
            observations["weak_negative_mask"],
        )
        geometry = factor_geometry_alignment_loss(
            output["factor_soft_masks"], observations["geometry_masks"], observations["geometry_known_mask"]
        )
        view = factor_view_consistency_loss(
            output["factor_presence_prob"], second_output_restored["factor_presence_prob"],
            output["factor_visibility_prob"], second_output_restored["factor_visibility_prob"],
            output["factor_soft_masks"], second_output_restored["factor_soft_masks"],
        )
        prototype = factor_prototype_regularization(
            output["prototype_weights"], model.factor_extractor.prototype_bank.prototypes,
            model.factor_extractor.prototype_bank.prototype_valid_mask, output["prior_scale"],
        )
        contradiction = factor_contradiction_consistency_loss(
            output["factor_presence_prob"], build_icdor_contradiction_mask(model).to(output["factor_presence_prob"].device)
        )
        anchor = factor_positive_anchor_loss(output["factor_presence_logits"], supervision)
        losses.update(factor)
        losses.update(view)
        losses.update(prototype)
        losses["loss_factor_geometry"] = geometry
        losses["loss_factor_contradiction"] = contradiction
        losses["loss_factor_positive_anchor"] = anchor
        factor_total = (
            float(factor_config.get("presence_weight", 1.0)) * factor["loss_factor_presence"]
            + float(factor_config.get("visibility_weight", 1.0)) * factor["loss_factor_visibility"]
            + 0.05 * factor["loss_factor_weak_negative"]
            + float(factor_config.get("geometry_mask_weight", 0.10)) * geometry
            + float(factor_config.get("view_consistency_weight", 0.05)) * view["loss_factor_view_probability"]
            + float(factor_config.get("flip_equivariance_weight", 0.05)) * view["loss_factor_flip_equivariance"]
            + float(factor_config.get("prototype_occupancy_weight", 0.02)) * prototype["loss_factor_prototype_occupancy"]
            + float(factor_config.get("prototype_repulsion_weight", 0.01)) * prototype["loss_factor_prototype_repulsion"]
            + float(factor_config.get("prior_scale_weight", 0.01)) * prototype["loss_factor_prior_scale"]
            + float(factor_config.get("contradiction_weight", 0.02)) * contradiction
            + float(factor_config.get("positive_anchor_weight", 1.0)) * anchor
        )
    observed_targets = reason_targets.masked_fill(hidden_mask, 0.0)
    observed_valid_mask = ~hidden_mask
    observed = reason_observed_losses(
        output["reason_visual_observed_logits"], output["reason_observed_logits"], observed_targets,
        observed_valid_mask=observed_valid_mask,
    )
    losses.update(observed)
    reason_total = (
        policy["reason_visual_observed"] * observed["loss_reason_visual_observed_asl"]
        if not phase.latent_enabled else
        policy["reason_visual_observed"] * observed["loss_reason_visual_observed_asl"]
        + float(reason_config.get("observed_asl_weight", 1.0)) * observed["loss_reason_observed_asl"]
    )
    posterior = model.observation_model.posterior_from_observed_targets(
        output["reason_logits_latent"], observed_targets, output
    )["reason_latent_posterior"]
    losses["reason_latent_posterior_mean"] = posterior.mean()
    if phase.latent_enabled and phase.pu_enabled:
        factor_support = torch.einsum(
            "brf,bf->br", output["reason_factor_router_weights"], output["factor_positive_evidence"].detach()
        )
        selective = selective_observation_losses(
            output["reason_logits_latent"], observed_targets,
            output["reason_observation_prob"], posterior,
            reason_observation_logits=output["reason_observation_logits"],
            reason_propensity=output["reason_propensity"], factor_route_support=factor_support,
            escape_weight=output["reason_escape_weight"], synthetic_hidden_positive_mask=hidden_mask,
            observed_valid_mask=observed_valid_mask,
        )
        losses.update(selective)
        reason_total = reason_total + (
            policy["reason_observation"] * selective["loss_reason_observation_nll"]
            + float(reason_config.get("posterior_bce_weight", 0.30)) * selective["loss_reason_posterior_bce"]
            + float(reason_config.get("posterior_rank_weight", 0.08)) * selective["loss_reason_posterior_rank"]
            + float(reason_config.get("factor_latent_consistency_weight", 0.05)) * selective["loss_reason_factor_latent_consistency"]
            + float(reason_config.get("escape_token_weight", 0.01)) * selective["loss_reason_escape"]
            + float(reason_config.get("propensity_regularization_weight", 0.01)) * selective["loss_reason_propensity"]
        )
        reason_rank, reason_rank_stats = posterior_weighted_reason_ranking_loss(
            output["reason_logits_latent"], posterior, sample_ids, reason_queue
        )
        losses["loss_reason_cross_sample_rank"] = reason_rank
        if phase.enable_posterior_ranking and float(reason_rank_stats["pair_weight_sum"]) > 0.0:
            reason_total = reason_total + policy["reason_posterior_rank"] * reason_rank
        losses["reason_rank_pair_weight_sum"] = reason_rank_stats["pair_weight_sum"]
    losses["action_rank_pair_weight_sum"] = action_rank_stats["pair_weight_sum"]
    losses.setdefault("loss_reason_cross_sample_rank", output["reason_logits_latent"].sum() * 0.0)
    losses.setdefault("reason_rank_pair_weight_sum", output["reason_logits_latent"].sum() * 0.0)
    losses["loss_action_total"] = action_total
    losses["loss_factor_weighted_total"] = factor_total
    losses["loss_reason_total"] = reason_total
    losses["loss_total"] = action_total + factor_total + reason_total
    if not torch.isfinite(losses["loss_total"]):
        raise FloatingPointError("IC-DOR produced a non-finite total loss")
    return losses


def _grounding_records(index: BDD100KGroundingIndex, file_names: list[str]) -> list[dict[str, Any] | None]:
    records: list[dict[str, Any] | None] = []
    for file_name in file_names:
        paths = index.lookup(file_name)
        if not paths.label_json and not paths.drivable_map:
            records.append(None)
            continue
        objects: list[dict[str, Any]] = []
        lanes: list[dict[str, Any]] = []
        if paths.label_json:
            try:
                payload = json.loads(Path(paths.label_json).read_text(encoding="utf-8", errors="ignore"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            for frame in payload.get("frames", []) or []:
                for item in [*(frame.get("objects", []) or []), *(frame.get("labels", []) or [])]:
                    if isinstance(item, dict) and "lane" in str(item.get("category", "")).lower():
                        lanes.append(item)
                    elif isinstance(item, dict):
                        objects.append(item)
        records.append({
            "image_size": (720, 1280),
            "objects": objects,
            "lanes": lanes,
            "drivable_map": paths.drivable_map,
        })
    return records


def _parameter_group_gradient_audit(
    losses: dict[str, torch.Tensor],
    ownership: list[dict[str, Any]],
    model: nn.Module,
    *,
    epoch: int,
    step: int,
) -> list[dict[str, Any]]:
    named = dict(model.named_parameters())
    groups: dict[str, list[nn.Parameter]] = {}
    for entry in ownership:
        parameter = named[entry["full_name"]]
        if parameter.requires_grad:
            groups.setdefault(entry["owner_group"], []).append(parameter)
    rows: list[dict[str, Any]] = []
    action_vectors: dict[str, torch.Tensor] = {}
    for loss_name in ("loss_action_total", "loss_factor_weighted_total", "loss_reason_total"):
        loss = losses[loss_name]
        for group_name, parameters in groups.items():
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            flat = torch.cat([gradient.detach().flatten() for gradient in gradients if gradient is not None], dim=0) if any(
                gradient is not None for gradient in gradients
            ) else loss.new_zeros(1)
            norm = flat.norm()
            row = {
                "epoch": epoch, "step": step, "loss": loss_name, "owner_group": group_name,
                "grad_norm": float(norm.cpu()), "finite": bool(torch.isfinite(norm)),
            }
            if loss_name == "loss_action_total":
                action_vectors[group_name] = flat
                row["cosine_with_action_base"] = 1.0 if norm > 0 else None
            else:
                action = action_vectors.get(group_name)
                row["cosine_with_action_base"] = (
                    float(torch.nn.functional.cosine_similarity(flat, action, dim=0).cpu())
                    if action is not None and flat.numel() == action.numel() and flat.norm() > 0 and action.norm() > 0 else None
                )
            rows.append(row)
    return rows


def assert_icdor_gradient_firewall(rows: list[dict[str, Any]], *, tolerance: float = 1e-7) -> None:
    """Reject cross-owner gradients that violate the v4 ownership contract."""
    forbidden = {
        "loss_action_total": {"reason_adapter", "reason_visual_decoder", "reason_latent_decoder", "reason_observed_mixer", "observation_model"},
        "loss_reason_total": {"visual_pyramid", "action_adapter", "action_visual_decoder", "action_router_rereader", "factor_extractor", "factor_prototypes"},
    }
    for row in rows:
        if row.get("loss") not in forbidden or row.get("owner_group") not in forbidden[row["loss"]]:
            continue
        value = float(row.get("grad_norm", 0.0))
        if value > tolerance:
            raise RuntimeError(
                f"IC-DOR gradient firewall violation: {row['loss']} reached {row['owner_group']} with norm {value:.3e}"
            )


def train_icdor_epoch(
    model: MOSAICTrustICDORModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    phase: ICDORPhase,
    config: dict[str, Any],
    ownership: list[dict[str, Any]],
    grounding_builder: ICDORGroundingObservationBuilder,
    grounding_index: BDD100KGroundingIndex,
    multiview: MOSAICWeakMultiView,
    pareto: MOSAICActionParetoAdmission,
    action_queue: MOSAICSoftRankQueue,
    reason_queue: MOSAICSoftRankQueue,
    gradient_accumulation_steps: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("IC-DOR gradient accumulation must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=device).manual_seed(int(config["data"]["split_seed"]) + epoch)
    action_queue_pending = MOSAICAccumulationQueueBuffer()
    reason_queue_pending = MOSAICAccumulationQueueBuffer()
    parameter_by_name = dict(model.named_parameters())
    owner_parameters: dict[str, list[nn.Parameter]] = {}
    for entry in ownership:
        parameter = parameter_by_name.get(str(entry["full_name"]))
        if parameter is not None:
            owner_parameters.setdefault(str(entry["owner_group"]), []).append(parameter)
    owner_clip_limits = {
        "visual_pyramid": 1.0,
        "action_visual_decoder": 1.0,
        "reason_visual_decoder": 1.0,
        "factor_adapter": 0.5,
        "factor_extractor": 0.5,
        "factor_prototypes": 0.5,
        "action_adapter": 1.0,
        "action_router_rereader": 0.5,
        "reason_adapter": 1.0,
        "reason_latent_decoder": 0.5,
        "reason_observed_mixer": 0.5,
        "observation_model": 0.5,
        "threshold_head": 1.0,
    }
    owner_clip_limits.update({str(k): float(v) for k, v in config["optimizer"].get("owner_clip", {}).items()})
    start = time.perf_counter()
    for step, cpu_batch in enumerate(loader):
        load_done = time.perf_counter()
        model.clear_batch_field_reuse()
        first_cpu, second_cpu, _, second_metadata = build_icdor_multiview_batch(cpu_batch["image"], multiview)
        first = first_cpu.to(device, non_blocking=True)
        second = second_cpu.to(device, non_blocking=True)
        batch = {
            "action": cpu_batch["action"].to(device, non_blocking=True),
            "reason": cpu_batch["reason"].to(device, non_blocking=True),
            "file_name": list(cpu_batch["file_name"]),
            # This loader is constructed exclusively from train_core indices.
            "split": ["train_core"] * len(cpu_batch["file_name"]),
        }
        observations = grounding_builder(
            _grounding_records(grounding_index, batch["file_name"]), device=device, split="train"
        )
        autocast_enabled = bool(config["training"]["bf16"] and device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            output = model(first, route_mode=phase.route_mode, latent_enabled=phase.latent_enabled, return_masks=True)
            second_output = model(second, route_mode="off", latent_enabled=False, return_masks=True)
            restored = {
                "factor_presence_prob": restore_factor_view(second_output["factor_presence_prob"], second_metadata, multiview, masks=False),
                "factor_visibility_prob": restore_factor_view(second_output["factor_visibility_prob"], second_metadata, multiview, masks=False),
                "factor_soft_masks": restore_factor_view(second_output["factor_soft_masks"], second_metadata, multiview, masks=True),
            }
            hidden = build_synthetic_hidden_positive_mask(
                batch["reason"],
                hide_fraction=float(config["selective_observation"]["synthetic_missing_positive_fraction"]),
                generator=generator,
            )
            losses = compute_icdor_training_losses(
                model, output, restored, batch, observations, phase, pareto, action_queue, reason_queue,
                hidden_mask=hidden, config=config,
            )
            scaled_loss = losses["loss_total"] / gradient_accumulation_steps
        with torch.no_grad():
            posterior = model.observation_model.posterior_from_observed_targets(
                output["reason_logits_latent"], batch["reason"].masked_fill(hidden, 0.0), output
            )["reason_latent_posterior"]
            action_queue_pending.add(output["action_visual_logits"], batch["action"], batch["file_name"])
            reason_queue_pending.add(output["reason_logits_latent"], posterior, batch["file_name"])
        if step % int(config["training"]["print_every"]) == 0:
            audit_rows = _parameter_group_gradient_audit(losses, ownership, model, epoch=epoch, step=step)
            assert_icdor_gradient_firewall(audit_rows)
            gradient_rows.extend(audit_rows)
        scaled_loss.backward()
        update = (step + 1) % gradient_accumulation_steps == 0 or step + 1 == len(loader)
        grad_norm = 0.0
        if update:
            owner_norms = clip_icdor_owner_gradients(owner_parameters, owner_clip_limits)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["optimizer"]["grad_clip"])).detach().cpu())
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            action_queue_pending.flush_after_optimizer_step(action_queue, optimizer_step_succeeded=True)
            reason_queue_pending.flush_after_optimizer_step(reason_queue, optimizer_step_succeeded=True)
            optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "step": step, "phase": phase.name, "update": update, "grad_norm": grad_norm}
        if update:
            row["owner_grad_norms"] = owner_norms
        row.update({name: float(value.detach().float().cpu()) for name, value in losses.items() if value.numel() == 1})
        rows.append(row)
        runtime_rows.append({
            "epoch": epoch, "step": step, "load_gap_sec": load_done - start,
            "step_sec": time.perf_counter() - load_done,
            "gpu_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0,
        })
        start = time.perf_counter()
        if step % int(config["training"]["print_every"]) == 0:
            print("icdor_batch " + json.dumps(row, sort_keys=True), flush=True)
    if not rows:
        raise RuntimeError("IC-DOR training loader produced no batches")
    return {"loss_rows": rows, "gradient_rows": gradient_rows, "runtime_rows": runtime_rows}


def fit_icdor_calibration(
    model: MOSAICTrustICDORModel,
    loader: DataLoader,
    device: torch.device,
    *,
    epoch: int,
    route_mode: str,
    latent_enabled: bool,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.threshold_head.parameters():
        parameter.requires_grad_(True)
    model.eval()
    action_logits: list[torch.Tensor] = []
    reason_logits: list[torch.Tensor] = []
    action_targets: list[torch.Tensor] = []
    reason_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            model.clear_batch_field_reuse()
            output = model(batch["image"].to(device, non_blocking=True), route_mode=route_mode, latent_enabled=latent_enabled)
            action_logits.append(output["action_logits_raw"].detach())
            reason_logits.append(output["reason_logits_raw"].detach())
            action_targets.append(batch["action"].to(device, non_blocking=True))
            reason_targets.append(batch["reason"].to(device, non_blocking=True))
    if not action_logits:
        raise RuntimeError("IC-DOR train_calib loader is empty")
    action_logit = torch.cat(action_logits)
    reason_logit = torch.cat(reason_logits)
    action_target = torch.cat(action_targets)
    reason_target = torch.cat(reason_targets)
    positive_support = torch.cat((action_target, reason_target), dim=1).sum(0).gt(0)
    unsupported_label_ids = torch.where(~positive_support)[0].detach().cpu().tolist()
    calibration = config["calibration"]
    optimizer = torch.optim.AdamW(model.threshold_head.parameters(), lr=float(config["optimizer"]["lr"]["threshold_head"]))
    rows: list[dict[str, Any]] = []
    batch_size = int(calibration["batch_size"])
    for step in range(int(calibration["steps_per_epoch"])):
        indices = (torch.arange(batch_size, device=device) + step * batch_size) % action_logit.shape[0]
        objective = model.threshold_head.calibration_objective(
            action_logit[indices], reason_logit[indices], action_target[indices], reason_target[indices],
            surrogate_temperature=float(calibration["surrogate_temperature"]),
            soft_f1_weight=float(calibration["soft_f1_weight"]), bce_weight=float(calibration["bce_weight"]),
            rate_weight=float(calibration["rate_weight"]), delta_weight=float(calibration["delta_weight"]),
            cardinality_weight=float(calibration["cardinality_weight"]),
            valid_label_mask=positive_support,
        )
        optimizer.zero_grad(set_to_none=True)
        objective["loss_calibration_total"].backward()
        optimizer.step()
        rows.append({
            "epoch": epoch, "step": step, "source_split": "train_calib",
            "positive_support_label_count": int(positive_support.sum().item()),
            "unsupported_label_ids": unsupported_label_ids,
            **{name: float(value.detach().cpu()) for name, value in objective.items() if value.numel() == 1},
        })
    return rows


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _validate_review_evidence_bindings(review_payload: Mapping[str, Any]) -> None:
    evidence = review_payload.get("evidence_files")
    if not isinstance(evidence, Mapping):
        raise RuntimeError("IC-DOR REVIEW_PASS evidence_files are missing")
    for name in (
        "pilot_gate", "factor_certificate", "edge_admission",
        "final_remediation_plan", "audit_addendum",
    ):
        binding = evidence.get(name)
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"IC-DOR REVIEW_PASS {name} binding is missing")
        path = Path(str(binding.get("path", "")))
        expected = str(binding.get("sha256", "")).upper()
        if not path.is_file() or not expected or _sha256_file(path) != expected:
            raise RuntimeError(f"IC-DOR REVIEW_PASS {name} evidence hash mismatch")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _visual_credibility_payload(result: Mapping[str, Any], *, epoch: int) -> dict[str, Any]:
    credibility = result.get("credibility")
    if not isinstance(credibility, torch.Tensor):
        raise ValueError("visual credibility refresh must return a tensor")
    return {
        "epoch": int(epoch),
        "source_split": result.get("source_split"),
        "reason_labels_used": result.get("reason_labels_used"),
        "factor_names": result.get("factor_names"),
        "credibility": credibility.detach().float().cpu().tolist(),
        "components": result.get("components"),
    }


def _audit_batches(
    loader: DataLoader,
    grounding_index: BDD100KGroundingIndex,
    *,
    source_split: str = "train_audit",
) -> Iterable[dict[str, Any]]:
    """Decorate disjoint audit rows with registered views and raw grounding records."""
    for batch in loader:
        images = batch["image"]
        # Photometric-only view remains pixel-registered. The explicit mirror
        # view is inverted by the collector before equivariance scoring.
        photometric = images * 0.97 + images.mean(dim=(-2, -1), keepdim=True) * 0.03
        yield {
            **batch,
            "split": [source_split] * images.shape[0],
            "grounding_records": _grounding_records(grounding_index, list(batch["file_name"])),
            "audit_views": torch.stack((images, photometric), dim=1),
            "audit_mirror_view": torch.flip(images, dims=(-1,)),
        }


def _candidate_edge_specs(ontology: dict[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for action_name, routes in ontology["action_routes"].items():
        for direction in ("support", "veto"):
            for edge in routes[direction]:
                specs.append({
                    "factor": str(edge["factor"]),
                    "action": str(action_name),
                    "direction": direction,
                    "polarity": str(edge["polarity"]),
                })
    return specs


def _target_transfer_directions(ontology: dict[str, Any]) -> tuple[list[list[str]], list[list[str]]]:
    """Return only ontology-authorized support/veto directions; all other pairs are absent."""
    factors = [str(item["name"]) for item in ontology["factors"]]
    factor_index = {name: index for index, name in enumerate(factors)}
    action = [["none" for _ in ontology["action_names"]] for _ in factors]
    reason = [["none" for _ in ontology["reason_names"]] for _ in factors]
    for action_name, routes in ontology["action_routes"].items():
        target = ontology["action_index"][action_name]
        for direction in ("support", "veto"):
            for edge in routes[direction]:
                action[factor_index[str(edge["factor"])]][target] = direction
    for target, routes in ontology["reason_routes"].items():
        for factor in (*routes["direct_factors"], *routes["latent_factors"]):
            reason[factor_index[str(factor)]][int(target)] = "support"
        for factor in routes["contradiction_factors"]:
            reason[factor_index[str(factor)]][int(target)] = "veto"
    return action, reason


@torch.no_grad()
def collect_action_pareto_audit(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    route_mode: str,
    latent_enabled: bool,
    source_split: str = "audit_target",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure visual and routed per-action AP on the target audit split."""
    if source_split not in {"audit_target", "train_audit"}:
        raise ValueError("Pareto audit source must be audit_target")
    visual_rows, routed_rows, labels = [], [], []
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            split = batch.get("split")
            split_values = [split] if isinstance(split, str) else list(split or [])
            if not split_values or any(value != source_split for value in split_values):
                raise ValueError(f"Pareto audit accepts {source_split} rows only")
            model.clear_batch_field_reuse()
            output = model(
                batch["image"].to(device), route_mode=route_mode,
                latent_enabled=latent_enabled, return_masks=False,
            )
            visual_rows.append(output["action_visual_logits"].detach().cpu())
            routed_rows.append(output["action_shadow_logits"].detach().cpu())
            labels.append(batch["action"].detach().cpu())
    finally:
        model.train(was_training)
    if not labels:
        raise RuntimeError(f"Pareto audit received no {source_split} rows")
    visual, routed, target = map(torch.cat, (visual_rows, routed_rows, labels))
    visual_ap, routed_ap = [], []
    for label in range(4):
        visual_ap.append(float(multilabel_metrics_from_logits(visual[:, label:label + 1], target[:, label:label + 1])["mAP"]))
        routed_ap.append(float(multilabel_metrics_from_logits(routed[:, label:label + 1], target[:, label:label + 1])["mAP"]))
    return torch.tensor(visual_ap, device=device), torch.tensor(routed_ap, device=device)


@torch.no_grad()
def collect_train_audit_branch_readiness(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    route_mode: str,
    latent_enabled: bool,
    seed: int,
    safe_joint_entry_exp_map: float | None,
    source_split: str = "audit_target",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect target utility/hidden-recovery evidence on the target audit split."""
    if source_split not in {"audit_target", "train_audit"}:
        raise ValueError("IC-DOR branch readiness source must be audit_target")
    tensors: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "action_visual", "action_shadow", "action_final", "action_labels",
            "reason_visual", "reason_final", "reason_shuffled", "reason_labels",
        )
    }
    hidden_fractions = (0.10, 0.30, 0.50)
    hidden_collections: dict[tuple[str, float], dict[str, list[torch.Tensor]]] = {
        (mode, fraction): {name: [] for name in ("posterior", "baseline", "observed", "hidden")}
        for mode in MISSINGNESS_MODES if mode != "null"
        for fraction in hidden_fractions
    }
    disallowed_max = 0.0
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        for batch in loader:
            split = batch.get("split")
            split_values = [split] if isinstance(split, str) else list(split or [])
            if not split_values or any(value != source_split for value in split_values):
                raise ValueError(f"IC-DOR readiness collector accepts {source_split} only")
            model.clear_batch_field_reuse()
            images = batch["image"].to(device, non_blocking=True)
            action_labels = batch["action"].to(device, non_blocking=True).float()
            reason_labels = batch["reason"].to(device, non_blocking=True).float()
            output = model(
                images, route_mode=route_mode, latent_enabled=latent_enabled,
                reason_route_mode="full", return_masks=False, return_diagnostics=True,
            )
            for destination, source in (
                ("action_visual", "action_visual_logits"),
                ("action_shadow", "action_shadow_logits"),
                ("action_final", "action_final_logits"),
                ("reason_visual", "reason_visual_observed_logits"),
                ("reason_final", "reason_observed_logits"),
                ("reason_shuffled", "reason_observed_logits_route_shuffled"),
            ):
                value = output.get(source)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"IC-DOR readiness collector is missing {source}")
                tensors[destination].append(value.detach().float().cpu())
            tensors["action_labels"].append(action_labels.cpu())
            tensors["reason_labels"].append(reason_labels.cpu())
            candidate = getattr(getattr(model, "action_router", None), "candidate_edge_mask", None)
            support, veto = output.get("support_weights"), output.get("veto_weights")
            if not isinstance(candidate, torch.Tensor) or not isinstance(support, torch.Tensor) or not isinstance(veto, torch.Tensor):
                raise ValueError("IC-DOR readiness collector requires candidate mask and route weights")
            disallowed_max = max(
                disallowed_max,
                float(support.masked_select(~candidate[0].to(device)).abs().max()) if bool((~candidate[0]).any()) else 0.0,
                float(veto.masked_select(~candidate[1].to(device)).abs().max()) if bool((~candidate[1]).any()) else 0.0,
            )
            posterior_fn = getattr(getattr(model, "observation_model", None), "posterior_from_observed_targets", None)
            propensity = output.get("reason_propensity")
            if not callable(posterior_fn) or not isinstance(propensity, torch.Tensor):
                raise ValueError("IC-DOR readiness collector requires the selective-observation posterior")
            for (mode, hide_fraction), collection in hidden_collections.items():
                hidden = build_hidden_mask(
                    reason_labels, mode=mode, hide_fraction=hide_fraction,
                    propensity=propensity, generator=generator,
                )
                observed = reason_labels.masked_fill(hidden, 0.0)
                posterior = posterior_fn(output["reason_logits_latent"], observed, output)["reason_latent_posterior"]
                collection["posterior"].append(posterior.detach().float().cpu())
                collection["baseline"].append(torch.sigmoid(output["reason_visual_observed_logits"]).detach().float().cpu())
                collection["observed"].append(observed.cpu())
                collection["hidden"].append(hidden.cpu())
    finally:
        model.train(was_training)
    if not tensors["action_labels"]:
        raise ValueError(f"IC-DOR readiness collector received no {source_split} rows")
    values = {name: torch.cat(rows) for name, rows in tensors.items()}
    def per_label_ap(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
        return [
            float(multilabel_metrics_from_logits(logits[:, index:index + 1], labels[:, index:index + 1])["mAP"])
            for index in range(logits.shape[1])
        ]
    action_visual_ap = per_label_ap(values["action_visual"], values["action_labels"])
    action_shadow_ap = per_label_ap(values["action_shadow"], values["action_labels"])
    action_final_ap = per_label_ap(values["action_final"], values["action_labels"])
    reason_visual_metrics = multilabel_metrics_from_logits(values["reason_visual"], values["reason_labels"])
    reason_final_metrics = multilabel_metrics_from_logits(values["reason_final"], values["reason_labels"])
    reason_shuffled_metrics = multilabel_metrics_from_logits(values["reason_shuffled"], values["reason_labels"])
    action_final_metrics = multilabel_metrics_from_logits(values["action_final"], values["action_labels"])
    route_delta = values["action_shadow"] - values["action_visual"]
    route_ratios = (
        route_delta.square().mean(0).sqrt()
        / values["action_visual"].square().mean(0).sqrt().clamp_min(1e-8)
    )
    hidden_rows: list[dict[str, Any]] = []
    for (mode, hide_fraction), collection in hidden_collections.items():
        hidden_rows.append(audit_hidden_recovery_scores(
            torch.cat(collection["posterior"]), torch.cat(collection["baseline"]),
            torch.cat(collection["observed"]), torch.cat(collection["hidden"]),
            mode=mode, hide_fraction=hide_fraction,
        ))
    available_hidden = [row for row in hidden_rows if row.get("available") is True]
    hidden_margin = min((float(row["margin"]) for row in available_hidden), default=-1.0)
    final_exp_map = float(reason_final_metrics["mAP"])
    return {
        "source_split": source_split,
        "action_shadow_ap_delta": [shadow - visual for shadow, visual in zip(action_shadow_ap, action_visual_ap)],
        "action_route_ap_delta": [final - visual for final, visual in zip(action_final_ap, action_visual_ap)],
        "exp_map_final": final_exp_map,
        "exp_map_delta_vs_visual": final_exp_map - float(reason_visual_metrics["mAP"]),
        "factor_shuffle_degrades_reason": final_exp_map > float(reason_shuffled_metrics["mAP"]),
        "hidden_recovery_margin": hidden_margin,
        "route_strength_ratio": float(route_ratios.mean()),
        "route_strength_ratio_by_action": route_ratios.tolist(),
        "disallowed_route_invariance": disallowed_max <= 1e-8,
        "disallowed_route_weight_max": disallowed_max,
        "train_audit_joint": 0.5 * (float(action_final_metrics["mF1"]) + float(reason_final_metrics["mF1"])),
        "exp_map_delta_vs_entry": (
            final_exp_map - safe_joint_entry_exp_map
            if safe_joint_entry_exp_map is not None else -1.0
        ),
    }, hidden_rows


def _save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_joint: float,
    certificate_sha256: str | None,
    edge_admission_sha256: str | None,
    config_sha256: str,
    split_sha256: str,
    action_queue: MOSAICSoftRankQueue,
    reason_queue: MOSAICSoftRankQueue,
    pareto: MOSAICActionParetoAdmission,
    adaptive_schedule: ICDORAdaptiveSchedule | None = None,
) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_joint": best_joint,
        "certificate_sha256": certificate_sha256,
        "edge_admission_sha256": edge_admission_sha256,
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "action_queue": action_queue.state_dict(),
        "reason_queue": reason_queue.state_dict(),
        "pareto": pareto.state_dict(),
        "adaptive_schedule": adaptive_schedule.state_dict() if adaptive_schedule is not None else None,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def _load_resume(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    certificate_path: Path,
    edge_path: Path,
    config_sha256: str,
    split_sha256: str,
    action_queue: MOSAICSoftRankQueue,
    reason_queue: MOSAICSoftRankQueue,
    pareto: MOSAICActionParetoAdmission,
    adaptive_schedule: ICDORAdaptiveSchedule | None = None,
) -> tuple[int, float, str | None, str | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != config_sha256 or payload.get("split_sha256") != split_sha256:
        raise RuntimeError("IC-DOR resume config/split hash mismatch")
    certificate_sha = payload.get("certificate_sha256")
    edge_sha = payload.get("edge_admission_sha256")
    if certificate_sha:
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        if certificate.get("sha256") != certificate_sha:
            raise RuntimeError("IC-DOR resume certificate hash mismatch")
        model.load_factor_certificate(certificate)
    if edge_sha:
        edge = json.loads(edge_path.read_text(encoding="utf-8"))
        if edge.get("sha256") != edge_sha:
            raise RuntimeError("IC-DOR resume edge-admission hash mismatch")
        entries = edge.get("entries", {})
        mask = torch.zeros_like(model.action_router.edge_admission_mask)
        accepted = [record for record in entries.values() if record.get("accepted")]
        if accepted:
            factor_index = model.ontology["factor_index"]
            action_index = model.ontology["action_index"]
            for record in accepted:
                direction = 0 if record["direction"] == "support" else 1
                mask[direction, factor_index[record["factor"]], action_index[record["target"]]] = True
        model.set_edge_admission(mask)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    for key in ("action_queue", "reason_queue", "pareto"):
        if key not in payload:
            raise RuntimeError(f"IC-DOR resume is missing {key} state")
    action_queue.load_state_dict(payload["action_queue"])
    reason_queue.load_state_dict(payload["reason_queue"])
    pareto.load_state_dict(payload["pareto"])
    if adaptive_schedule is not None:
        schedule_payload = payload.get("adaptive_schedule")
        if not isinstance(schedule_payload, dict):
            raise RuntimeError("IC-DOR resume is missing adaptive schedule state")
        adaptive_schedule.load_state_dict(schedule_payload)
    for key in ("python_rng_state", "torch_rng_state", "cuda_rng_state_all"):
        if key not in payload:
            raise RuntimeError(f"IC-DOR resume is missing {key}")
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return int(payload["epoch"]) + 1, float(payload.get("best_joint", float("-inf"))), certificate_sha, edge_sha


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    updates_per_epoch: int,
    warmup_epochs: int,
    min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(1, epochs * updates_per_epoch)
    warmup = max(1, warmup_epochs * updates_per_epoch)

    def factor(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-6)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _honest_row(epoch: int, artifact: str, reason: str) -> dict[str, Any]:
    return {"epoch": epoch, "artifact": artifact, "available": False, "reason": reason}


def _record_resolved_config_usage(
    resolved_tree: dict[str, Any], output: Path, main_config_name: str,
) -> dict[str, Any]:
    """Record per-leaf execution ownership without blanket-consume shortcuts."""
    tracker = ConfigUsageTracker(resolved_tree)
    runtime_sections = {
        "data": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "build_icdor_loaders"),
        "backbone": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "build_icdor_model"),
        "model": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "build_icdor_model/compute_icdor_training_losses"),
        "factor_certificate": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "adaptive_certificate_audit"),
        "edge_admission": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "adaptive_edge_audit"),
        "selective_observation": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "compute_icdor_training_losses"),
        "loss": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "compute_icdor_training_losses"),
        "optimizer": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "build_icdor_optimizer/_scheduler"),
        "training": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "train_icdor_epoch/main"),
        "calibration": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "fit_icdor_calibration"),
        "credibility": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "build_icdor_model/refresh_model_visual_credibility"),
        "fine_transport": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "build_icdor_model/evaluate_icdor"),
    }
    diagnostic_sections = {
        "experiment": ("fate_oia/engine/train_acpr_mosaic_trust_icdor.py", "load_config/run_manifest"),
        "evaluation": ("fate_oia/engine/eval_acpr_mosaic_trust_icdor.py", "evaluate_icdor"),
        "runtime": ("fate_oia/engine/profile_acpr_mosaic_trust_icdor.py", "runtime_profile_contract"),
    }
    prefix = main_config_name + "."
    for path in tracker.leaf_paths:
        if path.startswith(prefix):
            section = path[len(prefix):].split(".", 1)[0].split("[", 1)[0]
            if section in runtime_sections:
                file_name, symbol = runtime_sections[section]
                tracker.consume(path, consumer_file=file_name, consumer_symbol=symbol)
            elif section in diagnostic_sections:
                file_name, symbol = diagnostic_sections[section]
                tracker.diagnostic_only(path, consumer_file=file_name, consumer_symbol=symbol)
            else:
                raise ValueError(f"unclassified main config path: {path}")
    supplemental_consumers = {
        "mosaic_icdor_factor_candidates.yaml": "load_icdor_ontology/factor_measurement_and_supervision",
        "mosaic_icdor_action_routes.yaml": "load_icdor_ontology/MOSAICTargetSparseRouter",
        "mosaic_icdor_reason_routes.yaml": "load_icdor_ontology/MOSAICICDORLatentReasonDecoder",
        "mosaic_icdor_certificate_rules.yaml": "load_certificate_rules/build_factor_certificate",
    }
    for source_name, symbol in supplemental_consumers.items():
        tracker.consume_source(
            source_name, consumer_file="fate_oia/models/mosaic_native_semantics.py", consumer_symbol=symbol
        )
    payload = tracker.finalize(require_all_consumed=True)
    (output / "resolved_config_tree.json").write_text(
        json.dumps(resolved_tree, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolved_config_usage.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _build_rank_queues(*, capacity: int, device: torch.device) -> tuple[MOSAICSoftRankQueue, MOSAICSoftRankQueue]:
    return (
        MOSAICSoftRankQueue(4, capacity=capacity).to(device),
        MOSAICSoftRankQueue(21, capacity=capacity).to(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal ACPR-MOSAIC-TRUST V3 IC-DOR trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--runtime_selection")
    parser.add_argument("--review_pass")
    parser.add_argument("--require_review_pass", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_audit_samples", type=int)
    parser.add_argument("--max_calib_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if not args.pilot and not args.require_review_pass:
        raise RuntimeError("IC-DOR full training requires --require_review_pass")

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    semantic_sources = (
        config_path,
        Path("configs/mosaic_icdor_factor_candidates.yaml"),
        Path("configs/mosaic_icdor_action_routes.yaml"),
        Path("configs/mosaic_icdor_reason_routes.yaml"),
        Path("configs/mosaic_icdor_certificate_rules.yaml"),
    )
    resolved_tree = resolve_icdor_config_tree(semantic_sources)
    _record_resolved_config_usage(resolved_tree, output, config_path.name)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("IC-DOR requested CUDA but CUDA is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(config["training"]["tf32"])

    runtime_path = Path(args.runtime_selection or config["runtime"]["runtime_selection_path"])
    if not runtime_path.is_file():
        raise RuntimeError("IC-DOR requires a profiler-produced runtime selection")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("status") not in {"PASS", "selected"}:
        raise RuntimeError("IC-DOR runtime selection is not passing")
    batch_size = int(args.batch_size or runtime["batch_size"])
    grad_accum = int(args.gradient_accumulation_steps or runtime["grad_accum"])
    if args.require_review_pass:
        review = Path(args.review_pass or ".review/acpr_mosaic_trust_v3_icdor_REVIEW_PASS.json")
        if not review.is_file():
            raise RuntimeError("IC-DOR REVIEW_PASS is required before trainer execution")
        review_payload = json.loads(review.read_text(encoding="utf-8"))
        if review_payload.get("status") != "PASS" or review_payload.get("target_head") != _git_head():
            raise RuntimeError("IC-DOR REVIEW_PASS is stale or non-passing")
        if review_payload.get("resolved_config_sha256") != _sha256_file(config_path):
            raise RuntimeError("IC-DOR REVIEW_PASS config hash mismatch")
        if review_payload.get("runtime_selection_sha256") != _sha256_file(runtime_path):
            raise RuntimeError("IC-DOR REVIEW_PASS runtime hash mismatch")
        _validate_review_evidence_bindings(review_payload)
        gates = review_payload.get("gates", {})
        missing_gates = [name for name in ICDOR_REQUIRED_REMEDIATION_GATES if gates.get(name) != "PASS"]
        if missing_gates:
            raise RuntimeError(f"IC-DOR REVIEW_PASS lacks final remediation gates: {missing_gates}")
        current_tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if dirty or review_payload.get("target_tree") != current_tree:
            raise RuntimeError("IC-DOR REVIEW_PASS source tree is dirty or stale")
        tree_listing = subprocess.run(
            ["git", "ls-tree", "-r", "--full-tree", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout
        tree_manifest_sha = hashlib.sha256(tree_listing.encode("utf-8")).hexdigest().upper()
        if review_payload.get("source_manifest_sha256") != tree_manifest_sha:
            raise RuntimeError("IC-DOR REVIEW_PASS tracked source manifest mismatch")
        contract_manifest = Path(".review/icdor_source_manifest.json")
        if (
            not contract_manifest.is_file()
            or review_payload.get("contract_manifest_sha256") != _sha256_file(contract_manifest)
        ):
            raise RuntimeError("IC-DOR REVIEW_PASS contract source manifest mismatch")

    grounding_index = BDD100KGroundingIndex(config["data"]["bdd100k_root"])
    train_loader, audit_visual_loader, audit_target_loader, calib_loader, test_loader, split_stats = build_icdor_loaders(
        config, output, batch_size=batch_size, num_workers=args.num_workers,
        max_train_samples=args.max_train_samples, max_audit_samples=args.max_audit_samples,
        max_calib_samples=args.max_calib_samples, max_test_samples=args.max_test_samples,
        visual_grounding_index=grounding_index,
    )
    if args.require_review_pass and review_payload.get("split_sha256") != split_stats["split_sha256"]:
        raise RuntimeError("IC-DOR REVIEW_PASS split hash mismatch")
    model = build_icdor_model(config).to(device)
    optimizer, ownership = build_icdor_optimizer(model, config)
    (output / "parameter_ownership.json").write_text(
        json.dumps({"parameters": ownership}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    epochs = int(args.epochs or (6 if args.pilot else config["training"]["epochs"]))
    if args.pilot and epochs > 6:
        raise RuntimeError("IC-DOR pilot is limited to six epochs")
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    scheduler = _scheduler(
        optimizer, epochs=epochs, updates_per_epoch=updates_per_epoch,
        warmup_epochs=int(config["optimizer"]["warmup_epochs"]),
        min_ratio=float(config["optimizer"]["min_lr_ratio"]),
    )
    factor_names = [str(item["name"]) for item in model.ontology["factors"]]
    mirror_pairs = {name: name.replace("left", "right") for name in factor_names if "left" in name and name.replace("left", "right") in factor_names}
    mirror_pairs.update({value: key for key, value in tuple(mirror_pairs.items())})
    multiview = MOSAICWeakMultiView(factor_names, mirror_pairs=mirror_pairs, seed=args.seed)
    grounding_builder = ICDORGroundingObservationBuilder(model.ontology["factors"])
    pareto = MOSAICActionParetoAdmission()
    action_queue, reason_queue = _build_rank_queues(
        capacity=int(config["selective_observation"]["posterior_queue_size"]), device=device
    )
    adaptive_schedule = ICDORAdaptiveSchedule(pilot=args.pilot)

    certificate_path = output / "factor_certificate.json"
    edge_path = output / "edge_admission.json"
    pending_certificate = _pending_evidence_document("factor_certificate", build_epoch=int(config["factor_certificate"]["build_epoch"]))
    pending_edge = _pending_evidence_document("edge_admission", build_epoch=int(config["edge_admission"]["build_epoch"]))
    pretrained = Path(config["backbone"]["pretrained_weights"])
    manifest = {
        "command_line": [sys.executable, *sys.argv], "git_head": _git_head(),
        "credo_version": "v4_credo",
        "direct_image": True, "feature_cache": False, "token_compression": "none",
        "best_selection_split": "test", "best_selection_metric": "deploy_fixed_joint",
        "pretrained_weights": str(pretrained), "pretrained_sha256": _sha256_file(pretrained),
        "config_sha256": _sha256_file(config_path), "runtime_selection_sha256": _sha256_file(runtime_path),
        "batch_size": batch_size, "gradient_accumulation_steps": grad_accum,
        "effective_batch": batch_size * grad_accum, "pilot": bool(args.pilot),
    }
    source_manifest_path = Path(".review/icdor_source_manifest.json")
    if not source_manifest_path.is_file():
        raise RuntimeError("IC-DOR immutable source manifest is required")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    start_epoch, best_joint = 0, float("-inf")
    certificate_sha: str | None = None
    edge_sha: str | None = None
    if args.resume:
        start_epoch, best_joint, certificate_sha, edge_sha = _load_resume(
            args.resume, model=model, optimizer=optimizer, scheduler=scheduler,
            certificate_path=certificate_path, edge_path=edge_path,
            config_sha256=manifest["config_sha256"], split_sha256=split_stats["split_sha256"],
            action_queue=action_queue, reason_queue=reason_queue, pareto=pareto,
            adaptive_schedule=adaptive_schedule,
        )
    else:
        initialize_icdor_run_artifacts(
            output, manifest=manifest, config=config, source_manifest=source_manifest,
            split_manifest=split_stats, runtime_selection=runtime,
            factor_certificate=pending_certificate, edge_admission=pending_edge,
        )
    previous_certificate_tiers: dict[str, str] | None = None
    if certificate_sha is not None and certificate_path.is_file():
        previous_certificate_tiers = _certificate_tiers(
            json.loads(certificate_path.read_text(encoding="utf-8"))
        )
    # Bootstrap epoch 0 from the independent visual audit. Later refreshes
    # occur only after an epoch, so every training forward reads a detached
    # credibility state from the preceding audit pass.
    bootstrap_visual_audit = collect_factor_audit(
        model,
        _audit_batches(audit_visual_loader, grounding_index, source_split="audit_visual"),
        grounding_builder,
        factor_names=factor_names,
        device=device,
        bootstrap_replicates=int(config["credibility"]["bootstrap_replicates"]),
        bootstrap_seed=args.seed,
        forward_kwargs={"route_mode": "off", "latent_enabled": False, "return_masks": True},
        source_split="audit_visual",
    )
    bootstrap_credibility = refresh_model_visual_credibility(model, bootstrap_visual_audit)
    _write_json(output / "visual_credibility_bootstrap.json", _visual_credibility_payload(bootstrap_credibility, epoch=-1))
    for epoch in range(start_epoch, epochs):
        if adaptive_schedule.failed_closed:
            raise RuntimeError(f"IC-DOR adaptive schedule failed closed: {adaptive_schedule.failure_reason}")
        certificate_ready = certificate_sha is not None
        edge_ready = edge_sha is not None
        phase = _adaptive_phase(adaptive_schedule)
        adaptive_schedule.record_epoch_execution()
        _restore_phase_trainability(model, phase)
        route_gate_cap = apply_icdor_route_gate_schedule(model, phase)
        if adaptive_schedule.policy().freeze_propensity and adaptive_schedule.state_epochs == 0:
            apply_icdor_consolidation(model, optimizer, scheduler)
        epoch_train = train_icdor_epoch(
            model, train_loader, optimizer, device, epoch=epoch, phase=phase, config=config,
            ownership=ownership, grounding_builder=grounding_builder, grounding_index=grounding_index,
            multiview=multiview, pareto=pareto, action_queue=action_queue, reason_queue=reason_queue,
            gradient_accumulation_steps=grad_accum, scheduler=scheduler,
        )
        for row in epoch_train["runtime_rows"]:
            row["action_route_gate_cap"] = route_gate_cap

        factor_audit_payload = collect_factor_audit(
            model,
            _audit_batches(audit_visual_loader, grounding_index, source_split="audit_visual"),
            grounding_builder,
            factor_names=factor_names,
            device=device,
            bootstrap_replicates=(
                int(config["factor_certificate"]["bootstrap_replicates"])
                if adaptive_schedule.policy().write_provisional_certificate else 100
            ),
            bootstrap_seed=args.seed + epoch,
            forward_kwargs={"route_mode": "off", "latent_enabled": False, "return_masks": True},
            source_split="audit_visual",
        )
        visual_credibility = refresh_model_visual_credibility(model, factor_audit_payload)
        visual_credibility_payload = _visual_credibility_payload(visual_credibility, epoch=epoch)
        epoch_dir = output / f"epoch_{epoch:03d}"
        factor_audit_rows = _factor_audit_rows(factor_audit_payload, epoch=epoch)
        certificate_tiers_before_epoch = previous_certificate_tiers
        if _write_provisional_certificate_this_epoch(
            write_provisional=adaptive_schedule.policy().write_provisional_certificate,
            freeze_certificate=adaptive_schedule.policy().freeze_certificate,
        ):
            audit_stats_path = output / "factor_audit_stats.json"
            _write_json(audit_stats_path, factor_audit_payload)
            certificate = build_and_write_factor_certificate(audit_stats_path, certificate_path, config_root="configs")
            certificate_payload = certificate.to_dict()
            model.load_factor_certificate(certificate_payload)
            certificate_sha = certificate.sha256
            certificate_ready = True
            previous_certificate_tiers = _certificate_tiers(certificate_payload)
            (output / "factor_certificate_sha256.txt").write_text(certificate_sha + "\n", encoding="ascii")

        if adaptive_schedule.policy().enable_interventions and certificate_ready and not edge_ready:
            edge_payload = collect_edge_intervention_audit(
                model, _audit_batches(audit_target_loader, grounding_index, source_split="audit_target"), factor_names=factor_names,
                action_names=list(model.ontology["action_names"]), edge_specs=_candidate_edge_specs(model.ontology),
                device=device, bootstrap_replicates=1000, bootstrap_seed=args.seed,
                forward_kwargs={"route_mode": "admitted", "latent_enabled": True, "return_masks": False},
                source_split="audit_target",
            )
            _write_json(output / "edge_intervention_stats.json", edge_payload)
            tiers = [json.loads(certificate_path.read_text(encoding="utf-8"))["entries"][name]["tier"] for name in factor_names]
            admission = build_edge_admission(
                _edge_statistics_from_audit(edge_payload),
                model.ontology,
                tiers,
                source_split="audit_target",
                credibility_min=float(model.action_credibility_min_for_admission),
            )
            edge_document = admission.to_dict()
            _write_json(edge_path, edge_document)
            model.set_edge_admission(admission.edge_admission_mask.to(device))
            edge_sha = admission.sha256
            (output / "edge_admission_sha256.txt").write_text(edge_sha + "\n", encoding="ascii")

        edge_document = (
            json.loads(edge_path.read_text(encoding="utf-8"))
            if edge_path.is_file() else dict(pending_edge)
        )
        branch_readiness, hidden_recovery_rows = collect_train_audit_branch_readiness(
            model, _audit_batches(audit_target_loader, grounding_index, source_split="audit_target"), device,
            route_mode=phase.route_mode, latent_enabled=phase.latent_enabled,
            seed=args.seed + epoch,
            safe_joint_entry_exp_map=adaptive_schedule.safe_joint_entry_exp_map,
            source_split="audit_target",
        )

        calibration_rows = fit_icdor_calibration(
            model, calib_loader, device, epoch=epoch, route_mode=phase.route_mode,
            latent_enabled=phase.latent_enabled, config=config,
        )
        evaluation = evaluate_icdor(
            model, test_loader, device, epoch=epoch, route_mode=phase.route_mode,
            latent_enabled=phase.latent_enabled,
        )
        certificate_snapshot = (
            json.loads(certificate_path.read_text(encoding="utf-8"))
            if certificate_path.is_file() else dict(pending_certificate)
        )
        if adaptive_schedule.policy().enable_interventions:
            action_directions, reason_directions = _target_transfer_directions(model.ontology)
            transfer = collect_joint_target_transfer_metrics(
                model, _audit_batches(audit_target_loader, grounding_index, source_split="audit_target"),
                factor_ids=factor_names,
                action_ids=list(model.ontology["action_names"]),
                reason_ids=list(model.ontology["reason_names"]),
                action_directions=action_directions,
                reason_directions=reason_directions,
                device=device, route_mode=phase.route_mode, latent_enabled=phase.latent_enabled,
                intervention_chunk_size=int(config["runtime"]["target_transfer_intervention_chunk_size"]),
                source_split="audit_target",
            )
            transfer_summary = {
                "epoch": epoch, "available": True, "source_split": "audit_target",
                "schema_version": transfer["schema_version"],
                "collection_runtime": transfer["collection_runtime"],
                **transfer["summary"],
            }
            action_ids = {f"action:{name}" for name in model.ontology["action_names"]}
            transfer_rows = [
                {
                    "epoch": epoch, "available": True, "source_split": "audit_target",
                    "target_type": "action" if row["target_id"] in action_ids else "reason", **row,
                }
                for row in transfer["per_target"]
            ]
            target_utility_state = build_target_utility(
                {"source_split": "audit_target", "per_target": transfer["per_target"]},
                model.ontology,
            )
            model.target_utility.update_from_audit(
                torch.tensor(
                    target_utility_state["semantic_compatibility"], device=device, dtype=torch.float32
                ),
                torch.tensor(
                    target_utility_state["action_target_utility"], device=device, dtype=torch.float32
                ),
                source_split="audit_target",
            )
            semantic_compatibility_payload = {
                "schema_version": "mosaic_semantic_compatibility.v1",
                "epoch": epoch, "available": True, "source_split": "audit_target",
                "semantic_compatibility": target_utility_state["semantic_compatibility"],
                "per_target": target_utility_state["per_target"],
            }
            target_utility_payload = {
                "schema_version": "mosaic_target_utility.v1",
                "epoch": epoch, "available": True, "source_split": "audit_target",
                "action_target_utility": target_utility_state["action_target_utility"],
                "per_target": target_utility_state["per_target"],
            }
        else:
            transfer_summary = _honest_row(epoch, "target_transfer", "interventions_disabled_in_foundation")
            transfer_summary.update({"source_split": "audit_target", "phase": phase.name})
            transfer_rows = [dict(transfer_summary)]
            semantic_compatibility_payload = {
                "epoch": epoch,
                "available": False,
                "source_split": "audit_target",
                "unavailable_reason": "target_intervention_is_inactive_for_this_phase",
            }
            target_utility_payload = dict(semantic_compatibility_payload)
        epoch_dir = output / f"epoch_{epoch:03d}"
        visual_result = export_visual_audit(
            model, _audit_batches(audit_visual_loader, grounding_index, source_split="audit_visual"), epoch_dir,
            device=device, max_samples=16, source_split="audit_visual",
        )
        visual_manifest = json.loads(Path(visual_result["manifest"]).read_text(encoding="utf-8"))
        json_payloads = {
            "metrics_summary.json": evaluation["metrics_summary"],
            "branch_metrics.json": evaluation["branch_metrics"],
            "per_label_metrics.json": evaluation["per_label_metrics"],
            "factor_certificate_snapshot.json": certificate_snapshot,
            "target_transfer_summary.json": transfer_summary,
            "visual_credibility.json": visual_credibility_payload,
            "semantic_compatibility.json": semantic_compatibility_payload,
            "target_utility.json": target_utility_payload,
            "visual_audit_manifest.json": visual_manifest,
        }
        prototype_rows = evaluation["prototype_rows"]
        reason_rows = evaluation["reason_rows"] + [
            {"epoch": epoch, "source_split": "audit_target", "audit": "hidden_recovery", **row}
            for row in hidden_recovery_rows
        ]
        if phase.enable_pareto:
            visual_ap, routed_ap = collect_action_pareto_audit(
                model, _audit_batches(audit_target_loader, grounding_index, source_split="audit_target"), device,
                route_mode=phase.route_mode, latent_enabled=phase.latent_enabled,
                source_split="audit_target",
            )
            pareto_update = pareto.update_from_audit(visual_ap, routed_ap)
            pareto_rows = [{
                "epoch": epoch, "available": True, "source_split": "audit_target",
                "visual_ap": visual_ap.detach().cpu().tolist(),
                "routed_ap": routed_ap.detach().cpu().tolist(),
                "dual_variables": pareto.dual_variables.detach().cpu().tolist(),
                **pareto_update,
            }]
        else:
            pareto_rows = [_honest_row(epoch, "pareto", "pareto_is_inactive_for_this_phase")]
        train_core_readiness, train_audit_readiness, train_calib_readiness = _adaptive_readiness(
            epoch_train=epoch_train, factor_audit=factor_audit_payload, calibration_rows=calibration_rows,
            transfer_rows=transfer_rows, pareto_rows=pareto_rows, certificate_sha256=certificate_sha,
            edge_admission_sha256=edge_sha, certificate_snapshot=certificate_snapshot,
            previous_certificate_tiers=certificate_tiers_before_epoch, ontology=model.ontology,
            branch_readiness=branch_readiness, edge_document=edge_document,
            previous_best_train_audit_joint=adaptive_schedule.best_train_audit_joint,
        )
        transition = adaptive_schedule.update(
            epoch=epoch, train_core_metrics=train_core_readiness,
            train_audit_metrics=train_audit_readiness, train_calib_metrics=train_calib_readiness,
        )
        transition["certificate_sha256"] = certificate_sha
        transition["edge_admission_sha256"] = edge_sha
        current_train_audit_joint = float(branch_readiness["train_audit_joint"])
        adaptive_schedule.record_train_audit_reference(
            joint=current_train_audit_joint,
            exp_map=float(branch_readiness["exp_map_final"]),
            entered_safe_joint=(
                transition["state_before"] != "SAFE_JOINT"
                and transition["state_after"] == "SAFE_JOINT"
            ),
        )
        write_icdor_adaptive_schedule_transition(output, transition)
        if adaptive_schedule.failed_closed:
            raise RuntimeError(f"IC-DOR adaptive schedule failed closed: {adaptive_schedule.failure_reason}")
        jsonl_payloads = {
            "loss_components.jsonl": epoch_train["loss_rows"],
            "factor_stats.jsonl": factor_audit_rows,
            "prototype_stats.jsonl": prototype_rows,
            "action_route_stats.jsonl": evaluation["route_rows"],
            "reason_dual_observation_stats.jsonl": reason_rows,
            "target_transfer_stats.jsonl": transfer_rows,
            "pareto_stats.jsonl": pareto_rows,
            "gradient_ownership.jsonl": epoch_train["gradient_rows"],
            "calibration_stats.jsonl": calibration_rows,
            "runtime_stats.jsonl": epoch_train["runtime_rows"],
            "failure_cases.jsonl": evaluation["failure_rows"],
            "credibility_stats.jsonl": evaluation["credibility_rows"],
            "fine_transport_stats.jsonl": evaluation["fine_transport_rows"],
            "route_ownership.jsonl": evaluation["route_ownership_rows"],
        }
        write_icdor_epoch_artifacts(
            output, epoch=epoch, json_payloads=json_payloads, jsonl_payloads=jsonl_payloads,
            logits=evaluation["logits"], file_names=evaluation["file_names"],
        )
        joint = float(evaluation["metrics_summary"]["deploy_fixed"]["joint"])
        checkpoint_args = dict(
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            best_joint=max(best_joint, joint), certificate_sha256=certificate_sha,
            edge_admission_sha256=edge_sha, config_sha256=manifest["config_sha256"],
            split_sha256=split_stats["split_sha256"],
            action_queue=action_queue, reason_queue=reason_queue, pareto=pareto,
            adaptive_schedule=adaptive_schedule,
        )
        _save_checkpoint(output / "checkpoint_latest.pth", **checkpoint_args)
        if joint > best_joint:
            best_joint = joint
            _save_checkpoint(output / "checkpoint_best_test_joint.pth", **checkpoint_args)
        print("icdor_epoch " + json.dumps(evaluation["metrics_summary"], sort_keys=True), flush=True)

    if args.pilot:
        completed_epochs = list(range(start_epoch, epochs))
        schema = validate_icdor_artifact_schema(
            output, epochs=completed_epochs, strict_semantics=True, require_checkpoints=True
        )
        pending_artifacts = [] if schema.get("pass") else [
            *schema.get("missing", []), *schema.get("invalid", []), *schema.get("semantic_errors", [])
        ]
        final_certificate = (
            json.loads(certificate_path.read_text(encoding="utf-8"))
            if certificate_path.is_file() else {}
        )
        final_edge_document = (
            json.loads(edge_path.read_text(encoding="utf-8"))
            if edge_path.is_file() else {}
        )
        pilot_semantics = _pilot_semantic_validation(
            adaptive_schedule.state_dict(), final_certificate, final_edge_document
        )
        pilot_gate = {
            "git_head": _git_head(),
            "pass": (
                bool(schema.get("pass")) and pilot_semantics["pass"] is True
            ),
            "artifacts_complete": bool(schema.get("pass")),
            "pending_artifacts": pending_artifacts,
            "certificate_sha256": _sha256_file(certificate_path),
            "edge_admission_sha256": _sha256_file(edge_path),
            "certificate_semantic_sha256": certificate_sha,
            "edge_admission_semantic_sha256": edge_sha,
            "epochs": completed_epochs,
            "source_split": "train_audit",
            "learning_access_validated": pilot_semantics["pass"] is True,
            "deployment_admission_ready": pilot_semantics["deployment_admission_ready"],
            "adaptive_schedule": adaptive_schedule.state_dict(),
            "semantic_validation": pilot_semantics,
            "schema_validation": schema,
            "evidence_files": {
                "factor_certificate": {
                    "path": str(certificate_path.resolve()),
                    "sha256": _sha256_file(certificate_path),
                },
                "edge_admission": {
                    "path": str(edge_path.resolve()),
                    "sha256": _sha256_file(edge_path),
                },
            },
        }
        _write_json(output / "pilot_gate.json", pilot_gate)
        if pilot_gate["pass"] is not True:
            raise RuntimeError(f"IC-DOR pilot gate failed: {pilot_gate}")


if __name__ == "__main__":
    main()
