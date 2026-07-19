from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


ICDOR_ROOT_JSON_FILES = (
    "run_manifest.json",
    "source_manifest.json",
    "split_manifest.json",
    "runtime_selection.json",
    "factor_certificate.json",
    "edge_admission.json",
)
ICDOR_ROOT_JSONL_FILES = ("metrics_summary.jsonl", "adaptive_schedule.jsonl")
ICDOR_EPOCH_JSON_FILES = (
    "metrics_summary.json",
    "branch_metrics.json",
    "mechanism_summary.json",
    "per_label_metrics.json",
    "factor_certificate_snapshot.json",
    "factor_audit.json",
    "visual_credibility.json",
    "target_transfer_summary.json",
    "semantic_compatibility.json",
    "target_utility.json",
    "visual_audit_manifest.json",
)
ICDOR_EPOCH_JSONL_FILES = (
    "loss_components.jsonl",
    "factor_stats.jsonl",
    "prototype_stats.jsonl",
    "action_route_stats.jsonl",
    "reason_dual_observation_stats.jsonl",
    "target_transfer_stats.jsonl",
    "pareto_stats.jsonl",
    "gradient_ownership.jsonl",
    "calibration_stats.jsonl",
    "runtime_stats.jsonl",
    "failure_cases.jsonl",
    "credibility_stats.jsonl",
    "fine_transport_stats.jsonl",
    "route_ownership.jsonl",
)
ICDOR_LOGIT_FILES = (
    "action_visual_logits.pt",
    "action_shadow_logits.pt",
    "action_final_logits.pt",
    "action_deploy_logits.pt",
    "reason_visual_observed_logits.pt",
    "reason_latent_logits.pt",
    "reason_observation_model_prob.pt",
    "reason_observed_logits.pt",
    "reason_deploy_logits.pt",
    "action_factor_off_logits.pt",
    "action_factor_shuffled_logits.pt",
    "action_wrong_target_logits.pt",
    "action_equal_mass_random_logits.pt",
    "reason_factor_route_off_logits.pt",
    "reason_factor_route_shuffled_logits.pt",
    "action_labels.pt",
    "reason_labels.pt",
)

# CREDO assigns the three visual lanes to separate objectives.  These pairs
# must be present in every gradient audit and exactly zero, not merely absent.
ICDOR_REQUIRED_ZERO_GRADIENTS = {
    "loss_action_total": frozenset({
        "factor_visual_pyramid", "factor_adapter", "factor_extractor", "factor_prototypes",
        "reason_visual_pyramid", "reason_adapter", "reason_visual_decoder",
        "reason_latent_decoder", "reason_observed_mixer", "observation_model",
    }),
    "loss_reason_total": frozenset({
        "factor_visual_pyramid", "factor_adapter", "factor_extractor", "factor_prototypes",
        "action_visual_pyramid", "action_adapter", "action_visual_decoder", "action_router_rereader",
    }),
}


def _gradient_firewall_rows_valid(rows: list[dict[str, Any]], *, tolerance: float = 1e-7) -> bool:
    """Check that the persisted audit proves every CREDO cross-owner block."""
    observed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("loss")), str(row.get("owner_group")))
        observed.setdefault(key, []).append(row)
    for loss, owners in ICDOR_REQUIRED_ZERO_GRADIENTS.items():
        for owner in owners:
            candidates = observed.get((loss, owner), [])
            if not candidates:
                return False
            for candidate in candidates:
                try:
                    norm = float(candidate["grad_norm"])
                except (KeyError, TypeError, ValueError):
                    return False
                if candidate.get("finite") is not True or not torch.isfinite(torch.tensor(norm)) or abs(norm) > tolerance:
                    return False
    return True


def _mechanism_summary_valid(payload: Any) -> bool:
    """Validate the smoke-facing CREDO evidence summary is interpretable."""
    if not isinstance(payload, dict):
        return False
    required = {
        "schema_version", "epoch", "available", "missing_evidence",
        "continuous_credibility", "fine_transport", "reason_transport",
        "action_shadow", "pu", "target_effectiveness", "gradient_firewall",
        "interpretation",
    }
    if not required <= set(payload) or payload.get("schema_version") != "mosaic_icdor_mechanism_summary.v2":
        return False
    if not isinstance(payload.get("available"), bool) or not isinstance(payload.get("missing_evidence"), list):
        return False
    for section in (
        "continuous_credibility", "fine_transport", "reason_transport", "action_shadow",
        "pu", "target_effectiveness", "gradient_firewall",
    ):
        value = payload.get(section)
        if not isinstance(value, dict) or not isinstance(value.get("available"), bool):
            return False
    def _finite_number(section: dict[str, Any], key: str) -> bool:
        value = section.get(key)
        return isinstance(value, (int, float)) and bool(torch.isfinite(torch.tensor(float(value))))
    credibility = payload["continuous_credibility"]
    fine = payload["fine_transport"]
    reason = payload["reason_transport"]
    action = payload["action_shadow"]
    pu = payload["pu"]
    if not isinstance(credibility.get("content_beats_prior_factor_count"), int):
        return False
    if not all(_finite_number(fine, key) for key in (
        "fine_mask_delta_mean", "fine_off_action_shadow_delta_abs_mean",
        "fine_off_reason_latent_delta_abs_mean",
    )):
        return False
    if not all(_finite_number(reason, key) for key in (
        "route_off_logit_delta_abs_mean", "shuffle_logit_delta_abs_mean",
        "visual_exp_map", "final_exp_map",
    )):
        return False
    no_lane = reason.get("no_lane_absence_polarity")
    if not isinstance(no_lane, dict) or no_lane.get("available") is not True or no_lane.get("contract") != "observability_times_absence":
        return False
    if not all(_finite_number(action, key) for key in ("route_to_visual_rms_ratio_mean", "final_act_map")):
        return False
    if action.get("final_visual_exact") is not True:
        return False
    if not isinstance(pu.get("schedule_enabled"), bool) or not isinstance(payload["gradient_firewall"].get("pass"), bool):
        return False
    interpretation = payload.get("interpretation")
    return interpretation == {
        "learning_access": "continuous_credibility_and_shadow_routes",
        "deployment_admission": "edge_audit_only",
        "certificate_role": "final_reporting_only",
    }


def _safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"IC-DOR artifact cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    if not payload:
        raise ValueError(f"IC-DOR JSON artifact {path.name} must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    if not row:
        raise ValueError(f"IC-DOR JSONL artifact {path.name} must not contain an empty row")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_safe(row), sort_keys=True) + "\n")


def _nested_metric(branches: dict[str, Any], group: str, branch: str, metric: str) -> float | None:
    value = branches.get(group, {})
    value = value.get(branch, {}) if isinstance(value, dict) else {}
    candidate = value.get(metric) if isinstance(value, dict) else None
    return float(candidate) if isinstance(candidate, (int, float)) else None


def build_icdor_mechanism_summary(
    *,
    epoch: int,
    visual_credibility: dict[str, Any],
    branch_metrics: dict[str, Any],
    route_rows: list[dict[str, Any]],
    factor_rows: list[dict[str, Any]],
    factor_audit: dict[str, Any],
    fine_transport_rows: list[dict[str, Any]],
    route_ownership_rows: list[dict[str, Any]],
    reason_rows: list[dict[str, Any]],
    hidden_recovery_rows: list[dict[str, Any]],
    target_transfer_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    edge_admission: dict[str, Any],
    pu_enabled: bool,
) -> dict[str, Any]:
    """Summarize observable CREDO mechanism evidence without making an admission decision."""
    missing: list[str] = []
    if not isinstance(branch_metrics.get("action"), dict) or not isinstance(branch_metrics.get("reason"), dict):
        missing.append("branch_metrics")
    if not route_rows:
        missing.append("action_route_stats")
    if not factor_rows:
        missing.append("factor_stats")
    if factor_audit.get("source_split") != "audit_visual" or not isinstance(factor_audit.get("factor_stats"), dict):
        missing.append("factor_audit")
    if not fine_transport_rows:
        missing.append("fine_transport_stats")
    if not gradient_rows:
        missing.append("gradient_ownership")
    if not route_ownership_rows:
        missing.append("route_ownership")
    if not reason_rows:
        missing.append("reason_transport_rows")

    credibility = visual_credibility.get("credibility", [])
    if isinstance(credibility, torch.Tensor):
        credibility = credibility.detach().float().cpu().tolist()
    credibility_values = [float(value) for value in credibility if isinstance(value, (int, float))]
    per_action = [row for row in route_rows if row.get("summary") == "per_action_route_effect"]
    ratios = [float(row["route_to_visual_rms_ratio"]) for row in per_action if isinstance(row.get("route_to_visual_rms_ratio"), (int, float))]
    directions = [float(row["delta_gt_direction_agreement"]) for row in per_action if isinstance(row.get("delta_gt_direction_agreement"), (int, float))]
    support_nonzero = sum(float(row.get("support_delta_rms", 0.0)) > 1e-8 for row in per_action)
    veto_nonzero = sum(float(row.get("veto_delta_rms", 0.0)) > 1e-8 for row in per_action)
    route_credibility = [
        float(row["route_credibility_effective_mean"])
        for row in per_action
        if isinstance(row.get("route_credibility_effective_mean"), (int, float))
    ]
    fine_deltas = [float(row["fine_mask_delta_mean"]) for row in fine_transport_rows if isinstance(row.get("fine_mask_delta_mean"), (int, float))]
    fine_separation = [float(row["anchor_separation_mean"]) for row in fine_transport_rows if isinstance(row.get("anchor_separation_mean"), (int, float))]
    fine_action_shadow = [
        float(row["fine_off_action_shadow_delta_abs_mean"])
        for row in fine_transport_rows
        if isinstance(row.get("fine_off_action_shadow_delta_abs_mean"), (int, float))
    ]
    fine_reason_latent = [
        float(row["fine_off_reason_latent_delta_abs_mean"])
        for row in fine_transport_rows
        if isinstance(row.get("fine_off_reason_latent_delta_abs_mean"), (int, float))
    ]
    coarse_action_shadow = [
        float(row["coarse_off_action_shadow_delta_abs_mean"])
        for row in fine_transport_rows
        if isinstance(row.get("coarse_off_action_shadow_delta_abs_mean"), (int, float))
    ]
    coarse_reason_latent = [
        float(row["coarse_off_reason_latent_delta_abs_mean"])
        for row in fine_transport_rows
        if isinstance(row.get("coarse_off_reason_latent_delta_abs_mean"), (int, float))
    ]
    hidden_margins = [float(row["margin"]) for row in hidden_recovery_rows if row.get("available") is True and isinstance(row.get("margin"), (int, float))]
    transfer = [row for row in target_transfer_rows if row.get("available") is True]
    accepted = [row for row in (edge_admission.get("entries") or {}).values() if isinstance(row, dict) and row.get("accepted") is True]
    factor_stats = factor_audit.get("factor_stats", {})
    content_beats_prior = sum(
        isinstance(stats, dict)
        and isinstance(stats.get("scores"), dict)
        and isinstance(stats["scores"].get("content_only"), (int, float))
        and isinstance(stats["scores"].get("prior_only"), (int, float))
        and float(stats["scores"]["content_only"]) > float(stats["scores"]["prior_only"])
        for stats in factor_stats.values()
    )
    final_visual_exact = bool(route_ownership_rows) and all(
        row.get("action_final_visual_equal") is True for row in route_ownership_rows
    )
    route_off_logit_deltas = [
        float(row["factor_route_effect_abs_mean"])
        for row in reason_rows
        if isinstance(row.get("factor_route_effect_abs_mean"), (int, float))
    ]
    shuffle_logit_deltas = [
        float(row["factor_shuffle_effect_abs_mean"])
        for row in reason_rows
        if isinstance(row.get("factor_shuffle_effect_abs_mean"), (int, float))
    ]
    no_lane_rows = [
        row for row in reason_rows
        if int(row.get("reason_id", -1)) in {9, 15}
    ]
    no_lane_absence_mass = [
        float(row["absence_factor_mass_mean"])
        for row in no_lane_rows
        if isinstance(row.get("absence_factor_mass_mean"), (int, float))
    ]
    no_lane_negative_evidence = [
        float(row["absence_negative_evidence_mean"])
        for row in no_lane_rows
        if isinstance(row.get("absence_negative_evidence_mean"), (int, float))
    ]

    visual_reason = _nested_metric(branch_metrics, "reason", "visual_observed", "Exp_mAP")
    final_reason = _nested_metric(branch_metrics, "reason", "final_observed", "Exp_mAP")
    route_off_reason = _nested_metric(branch_metrics, "reason", "factor_route_off", "Exp_mAP")
    shuffled_reason = _nested_metric(branch_metrics, "reason", "factor_route_shuffled", "Exp_mAP")
    visual_action = _nested_metric(branch_metrics, "action", "visual", "Act_mAP")
    shadow_action = _nested_metric(branch_metrics, "action", "shadow", "Act_mAP")
    final_action = _nested_metric(branch_metrics, "action", "final", "Act_mAP")

    return {
        "schema_version": "mosaic_icdor_mechanism_summary.v2",
        "epoch": int(epoch),
        "available": not missing,
        "missing_evidence": missing,
        "continuous_credibility": {
            "available": bool(credibility_values),
            "nonzero_factor_count": sum(value > 1e-8 for value in credibility_values),
            "observable_cV_gt_030_count": sum(value > 0.30 for value in credibility_values),
            "content_beats_prior_factor_count": int(content_beats_prior),
            "mean": float(sum(credibility_values) / len(credibility_values)) if credibility_values else None,
            "mean_abs_delta": visual_credibility.get("mean_abs_cV_delta"),
        },
        "fine_transport": {
            "available": bool(fine_deltas),
            "fine_mask_delta_mean": float(sum(fine_deltas) / len(fine_deltas)) if fine_deltas else None,
            "anchor_separation_mean": float(sum(fine_separation) / len(fine_separation)) if fine_separation else None,
            "fine_off_action_shadow_delta_abs_mean": float(sum(fine_action_shadow) / len(fine_action_shadow)) if fine_action_shadow else None,
            "fine_off_reason_latent_delta_abs_mean": float(sum(fine_reason_latent) / len(fine_reason_latent)) if fine_reason_latent else None,
            "coarse_off_action_shadow_delta_abs_mean": float(sum(coarse_action_shadow) / len(coarse_action_shadow)) if coarse_action_shadow else None,
            "coarse_off_reason_latent_delta_abs_mean": float(sum(coarse_reason_latent) / len(coarse_reason_latent)) if coarse_reason_latent else None,
        },
        "reason_transport": {
            "available": all(value is not None for value in (visual_reason, final_reason, route_off_reason, shuffled_reason)),
            "visual_exp_map": visual_reason,
            "final_exp_map": final_reason,
            "route_off_delta_exp_map": (final_reason - route_off_reason) if final_reason is not None and route_off_reason is not None else None,
            "shuffle_delta_exp_map": (final_reason - shuffled_reason) if final_reason is not None and shuffled_reason is not None else None,
            "route_off_logit_delta_abs_mean": float(sum(route_off_logit_deltas) / len(route_off_logit_deltas)) if route_off_logit_deltas else None,
            "shuffle_logit_delta_abs_mean": float(sum(shuffle_logit_deltas) / len(shuffle_logit_deltas)) if shuffle_logit_deltas else None,
            "no_lane_absence_polarity": {
                "available": len(no_lane_absence_mass) == 2 and len(no_lane_negative_evidence) == 2,
                "contract": "observability_times_absence",
                "absence_factor_mass_mean": float(sum(no_lane_absence_mass) / len(no_lane_absence_mass)) if no_lane_absence_mass else None,
                "negative_evidence_mean": float(sum(no_lane_negative_evidence) / len(no_lane_negative_evidence)) if no_lane_negative_evidence else None,
            },
        },
        "action_shadow": {
            "available": bool(ratios) and visual_action is not None and shadow_action is not None and final_action is not None,
            "visual_act_map": visual_action,
            "shadow_act_map": shadow_action,
            "final_act_map": final_action,
            "final_visual_exact": final_visual_exact,
            "shadow_minus_visual_act_map": (shadow_action - visual_action) if shadow_action is not None and visual_action is not None else None,
            "route_to_visual_rms_ratio_mean": float(sum(ratios) / len(ratios)) if ratios else None,
            "delta_gt_direction_agreement_mean": float(sum(directions) / len(directions)) if directions else None,
            "support_nonzero_action_count": support_nonzero,
            "veto_nonzero_action_count": veto_nonzero,
            "route_credibility_effective_mean": float(sum(route_credibility) / len(route_credibility)) if route_credibility else None,
        },
        "pu": {
            "available": bool(hidden_margins),
            "hidden_recovery_margin_min": min(hidden_margins) if hidden_margins else None,
            "enabled_by_margin": bool(hidden_margins) and min(hidden_margins) > 0.0,
            "schedule_enabled": bool(pu_enabled),
        },
        "target_effectiveness": {
            "available": bool(transfer),
            "tet_mean": float(sum(float(row.get("tet", 0.0)) for row in transfer) / len(transfer)) if transfer else None,
            "tes_mean": float(sum(float(row.get("tes", 0.0)) for row in transfer) / len(transfer)) if transfer else None,
            "cca_mean": float(sum(float(row.get("cca", 0.0)) for row in transfer) / len(transfer)) if transfer else None,
            "accepted_edge_count": len(accepted),
        },
        "gradient_firewall": {
            "available": bool(gradient_rows),
            "pass": _gradient_firewall_rows_valid(gradient_rows),
        },
        "interpretation": {
            "learning_access": "continuous_credibility_and_shadow_routes",
            "deployment_admission": "edge_audit_only",
            "certificate_role": "final_reporting_only",
        },
    }


def validate_icdor_pilot_mechanism(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate CREDO learning access without demanding deployment admission.

    The 6-epoch pilot exists to prove that visual factor measurement, latent
    reason transport, and the shadow action route learn useful signals while
    final action remains exactly visual. It must not require an edge or a
    certificate, which are deployment/reporting decisions made later.
    """
    errors: list[str] = []
    ordered = sorted((row for row in summaries if isinstance(row, dict)), key=lambda row: int(row.get("epoch", -1)))
    if len(ordered) < 2:
        errors.append("pilot_requires_at_least_two_epoch_summaries")
        return {"pass": False, "errors": errors, "epoch_count": len(ordered)}
    if any(row.get("available") is not True for row in ordered):
        errors.append("mechanism_summary_has_missing_evidence")
    if any(row.get("gradient_firewall", {}).get("pass") is not True for row in ordered):
        errors.append("gradient_firewall_failed")
    if any(row.get("action_shadow", {}).get("final_visual_exact") is not True for row in ordered):
        errors.append("final_action_is_not_exactly_visual_during_pilot")

    later = ordered[1:]
    if not any(float(row.get("action_shadow", {}).get("route_to_visual_rms_ratio_mean") or 0.0) >= 0.005 for row in later):
        errors.append("shadow_route_never_reaches_minimum_effect")
    if not any(
        float(row.get("reason_transport", {}).get("route_off_logit_delta_abs_mean") or 0.0) > 1e-8
        and float(row.get("reason_transport", {}).get("shuffle_logit_delta_abs_mean") or 0.0) > 1e-8
        for row in later
    ):
        errors.append("reason_full_off_shuffle_do_not_differ")
    if any(
        row.get("reason_transport", {}).get("no_lane_absence_polarity", {}).get("available") is not True
        or row.get("reason_transport", {}).get("no_lane_absence_polarity", {}).get("contract") != "observability_times_absence"
        for row in later
    ):
        errors.append("no_lane_absence_polarity_diagnostic_is_missing")
    if not any(
        float(row.get("fine_transport", {}).get("fine_off_action_shadow_delta_abs_mean") or 0.0) > 1e-8
        and float(row.get("fine_transport", {}).get("fine_off_reason_latent_delta_abs_mean") or 0.0) > 1e-8
        for row in later
    ):
        errors.append("fine_coordinate_shuffle_does_not_affect_shadow_or_latent_reason")
    if not any(
        int(row.get("action_shadow", {}).get("support_nonzero_action_count") or 0) >= 1
        and int(row.get("action_shadow", {}).get("veto_nonzero_action_count") or 0) >= 1
        for row in later
    ):
        errors.append("support_or_veto_route_is_inactive")

    initial_action = ordered[0].get("action_shadow", {}).get("final_act_map")
    final_action = ordered[-1].get("action_shadow", {}).get("final_act_map")
    if not isinstance(initial_action, (int, float)) or not isinstance(final_action, (int, float)) or float(final_action) <= float(initial_action):
        errors.append("raw_action_map_did_not_improve_over_epoch_zero")
    final_reason = ordered[-1].get("reason_transport", {}).get("final_exp_map")
    visual_reason = ordered[-1].get("reason_transport", {}).get("visual_exp_map")
    if not isinstance(final_reason, (int, float)) or not isinstance(visual_reason, (int, float)) or float(final_reason) < float(visual_reason) - 0.005:
        errors.append("final_reason_regressed_vs_visual")
    initial_reason = ordered[0].get("reason_transport", {}).get("final_exp_map")
    if not isinstance(initial_reason, (int, float)) or not isinstance(final_reason, (int, float)) or float(final_reason) <= float(initial_reason):
        errors.append("raw_reason_map_did_not_improve_over_epoch_zero")
    initial_content = int(ordered[0].get("continuous_credibility", {}).get("content_beats_prior_factor_count") or 0)
    final_content = int(ordered[-1].get("continuous_credibility", {}).get("content_beats_prior_factor_count") or 0)
    if final_content <= initial_content:
        errors.append("content_only_factor_quality_did_not_grow")
    for row in later:
        pu = row.get("pu", {})
        if pu.get("available") is not True:
            errors.append("pu_margin_is_unavailable")
            break
        if bool(pu.get("enabled_by_margin")) != bool(pu.get("schedule_enabled")):
            errors.append("pu_margin_does_not_control_only_pu_route")
            break
    return {
        "pass": not errors,
        "errors": errors,
        "epoch_count": len(ordered),
        "initial_action_map": initial_action,
        "final_action_map": final_action,
        "initial_reason_map": initial_reason,
        "final_reason_map": final_reason,
        "initial_content_beats_prior_factor_count": initial_content,
        "final_content_beats_prior_factor_count": final_content,
    }


def initialize_icdor_run_artifacts(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    source_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    runtime_selection: dict[str, Any],
    factor_certificate: dict[str, Any],
    edge_admission: dict[str, Any],
) -> Path:
    """Write immutable run provenance before the first optimizer step."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "run_manifest.json", manifest)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(_safe(config), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(output / "source_manifest.json", source_manifest)
    _write_json(output / "split_manifest.json", split_manifest)
    _write_json(output / "runtime_selection.json", runtime_selection)
    _write_json(output / "factor_certificate.json", factor_certificate)
    _write_json(output / "edge_admission.json", edge_admission)
    _append_jsonl(output / "adaptive_schedule.jsonl", {
        "event": "initialized", "state_after": "JOINT_SHADOW", "state_epochs_after": 0,
        "source_splits": ["train_core", "audit_visual", "audit_target", "train_calib"],
    })
    return output


def write_icdor_adaptive_schedule_transition(
    output_dir: str | Path,
    transition: dict[str, Any],
) -> Path:
    """Append a source-sealed adaptive state transition for later audit."""
    required = {
        "epoch", "state_before", "state_after", "state_epochs_before", "state_epochs_after",
        "ready", "failed_closed", "readiness", "certificate_sha256", "edge_admission_sha256",
    }
    if not required <= set(transition):
        raise ValueError("IC-DOR adaptive schedule transition is incomplete")
    readiness = transition["readiness"]
    if not isinstance(readiness, dict) or set(readiness) != {"train_core", "train_audit", "train_calib"}:
        raise ValueError("IC-DOR adaptive schedule readiness must contain only train_core/train_audit/train_calib")
    for split, metrics in readiness.items():
        if not isinstance(metrics, dict) or metrics.get("source_split") != split:
            actual = metrics.get("source_split") if isinstance(metrics, dict) else type(metrics).__name__
            raise ValueError(
                f"IC-DOR adaptive schedule readiness for {split} has invalid provenance: {actual}"
            )
        if any(value == "test" for value in metrics.values()):
            raise ValueError("IC-DOR adaptive schedule artifacts must not contain test provenance")
    path = Path(output_dir) / "adaptive_schedule.jsonl"
    _append_jsonl(path, transition)
    return path


def write_icdor_epoch_artifacts(
    output_dir: str | Path,
    *,
    epoch: int,
    json_payloads: dict[str, dict[str, Any]],
    jsonl_payloads: dict[str, list[dict[str, Any]]],
    logits: dict[str, torch.Tensor],
    file_names: list[str],
) -> Path:
    """Fail closed unless every IC-DOR interpretation surface is populated."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("IC-DOR epoch must be a non-negative integer")
    if set(json_payloads) != set(ICDOR_EPOCH_JSON_FILES):
        raise ValueError("IC-DOR epoch JSON schema is incomplete or contains an unknown artifact")
    if set(jsonl_payloads) != set(ICDOR_EPOCH_JSONL_FILES):
        raise ValueError("IC-DOR epoch JSONL schema is incomplete or contains an unknown artifact")
    if set(logits) != set(ICDOR_LOGIT_FILES):
        raise ValueError("IC-DOR epoch logits schema is incomplete or contains an unknown artifact")
    sample_count = len(file_names)
    if sample_count <= 0 or any(not isinstance(name, str) or not name for name in file_names):
        raise ValueError("IC-DOR epoch file_names must be non-empty strings")
    for name, tensor in logits.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] != sample_count:
            raise ValueError(f"IC-DOR {name} must align to file_names")
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"IC-DOR {name} is empty or non-finite")
    output = Path(output_dir)
    epoch_dir = output / f"epoch_{epoch:03d}"
    logits_dir = epoch_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in json_payloads.items():
        _write_json(epoch_dir / name, payload)
    for name, rows in jsonl_payloads.items():
        if not rows:
            raise ValueError(f"IC-DOR {name} requires at least one real diagnostic row")
        for row in rows:
            _append_jsonl(epoch_dir / name, row)
    for name, tensor in logits.items():
        torch.save(tensor.detach().cpu(), logits_dir / name)
    _write_json(logits_dir / "file_names.json", file_names)
    _append_jsonl(output / "metrics_summary.jsonl", json_payloads["metrics_summary.json"])
    return epoch_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _matched_control_provenance_valid(arms: Any) -> bool:
    if not isinstance(arms, list) or len(arms) < 4:
        return False
    identity = [arm for arm in arms if arm.get("control_type") == "same_type_identity"]
    spatial = [arm for arm in arms if arm.get("control_type") == "spatial_roll"]
    if len(identity) != 1 or len(spatial) < 3:
        return False
    if any(int(arm.get("available_sample_count", 0)) <= 0 for arm in arms):
        return False
    identity_arm = identity[0]
    identity_names = identity_arm.get("identity_source_factor_names")
    identity_types = identity_arm.get("identity_source_factor_types")
    identity_regions = identity_arm.get("identity_source_regions")
    identity_sources = identity_arm.get("identity_sources")
    selected_type = identity_arm.get("factor_type")
    selected_region = identity_arm.get("region")
    selected_indices = identity_arm.get("selected_factor_indices")
    if (
        not isinstance(identity_names, list) or not identity_names
        or not isinstance(identity_types, list) or not identity_types
        or not isinstance(identity_regions, list) or not identity_regions
        or not isinstance(identity_sources, list) or not identity_sources
        or not selected_type or selected_region not in {
            "upper_front", "front_center", "left_corridor", "right_corridor", "center_corridor"
        }
        or not isinstance(selected_indices, list) or len(selected_indices) != 1
        or not isinstance(selected_indices[0], int)
        or any(
            not isinstance(source, dict)
            or not isinstance(source.get("index"), int)
            or not isinstance(source.get("name"), str) or not source["name"]
            or source.get("type") != selected_type
            or source.get("region") != selected_region
            or source["index"] == selected_indices[0]
            for source in identity_sources
        )
        or identity_arm.get("identity_source_factor_indices") != [source["index"] for source in identity_sources]
        # Display names are deliberately sorted independently in the compact
        # summary.  Their semantic pairing is carried by identity_sources.
        or sorted(identity_names) != sorted(source["name"] for source in identity_sources)
        or sorted(identity_types) != sorted({source["type"] for source in identity_sources})
        or sorted(identity_regions) != sorted({source["region"] for source in identity_sources})
    ):
        return False
    for arm in spatial:
        offsets = arm.get("spatial_offsets")
        if not isinstance(offsets, list) or not offsets:
            return False
        if any(
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or not all(isinstance(value, int) for value in offset)
            or tuple(offset) == (0, 0)
            for offset in offsets
        ):
            return False
        if arm.get("factor_type") != selected_type or arm.get("region") != selected_region:
            return False
    return all(
        arm.get("max_mass_error") is not None
        and float(arm["max_mass_error"]) <= 0.05
        and arm.get("max_overlap") is not None
        and float(arm["max_overlap"]) == 0.0
        and arm.get("control_support_method") == "topk_continuous_evidence"
        and isinstance(arm.get("control_evidence_slots"), int)
        and int(arm["control_evidence_slots"]) > 0
        and isinstance(arm.get("selected_support_count_mean"), (int, float))
        and float(arm["selected_support_count_mean"]) > 0.0
        and isinstance(arm.get("selected_mass_fraction_mean"), (int, float))
        and 0.0 < float(arm["selected_mass_fraction_mean"]) <= 1.0
        and isinstance(arm.get("source_region_mass_total"), (int, float))
        and float(arm["source_region_mass_total"]) > 0.0
        and arm.get("selected_factor_indices") == selected_indices
        for arm in arms
    )


def validate_icdor_artifact_schema(
    output_dir: str | Path,
    *,
    epochs: list[int],
    strict_semantics: bool = False,
    require_checkpoints: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    missing: list[str] = []
    invalid: list[str] = []
    for name in (*ICDOR_ROOT_JSON_FILES, "resolved_config.yaml", *ICDOR_ROOT_JSONL_FILES):
        path = output / name
        if not path.exists():
            missing.append(str(path))
        elif not path.stat().st_size:
            invalid.append(str(path))
    if require_checkpoints:
        for name in ("checkpoint_latest.pth", "checkpoint_best_test_joint.pth"):
            path = output / name
            if not path.exists() or not path.stat().st_size:
                missing.append(str(path))
    for epoch in epochs:
        epoch_dir = output / f"epoch_{epoch:03d}"
        for name in (*ICDOR_EPOCH_JSON_FILES, *ICDOR_EPOCH_JSONL_FILES):
            path = epoch_dir / name
            if not path.exists():
                missing.append(str(path))
            elif not path.stat().st_size:
                invalid.append(str(path))
        for name in (*ICDOR_LOGIT_FILES, "file_names.json"):
            path = epoch_dir / "logits" / name
            if not path.exists():
                missing.append(str(path))
            elif not path.stat().st_size:
                invalid.append(str(path))
    result: dict[str, Any] = {"pass": not missing and not invalid, "missing": missing, "invalid": invalid}
    if not strict_semantics or missing or invalid:
        return result
    errors: list[str] = []
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    for key, expected in (("direct_image", True), ("feature_cache", False), ("token_compression", "none"), ("best_selection_split", "test")):
        if manifest.get(key) != expected:
            errors.append(f"run_manifest.{key} must equal {expected!r}")
    if not manifest.get("git_head") or not manifest.get("pretrained_sha256"):
        errors.append("run_manifest must retain git_head and pretrained_sha256")
    v5_run = manifest.get("credo_version") == "v5_credo_map"
    split_manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    if v5_run:
        all_names = split_manifest.get("file_names")
        visual_ids = split_manifest.get("audit_visual_indices")
        target_ids = split_manifest.get("audit_target_indices")
        core_ids = split_manifest.get("train_core_indices")
        calib_ids = split_manifest.get("train_calib_indices")
        if not all(isinstance(value, list) for value in (all_names, visual_ids, target_ids, core_ids, calib_ids)):
            errors.append("split_manifest must expose list-valued core/audit/calibration partitions")
        else:
            total = len(all_names)
            expected_visual = max(1, round(total * 0.05))
            expected_target = max(1, round(total * 0.05))
            if len(visual_ids) != expected_visual or len(target_ids) != expected_target:
                errors.append("split_manifest audit_visual/audit_target must each be exactly 5 percent")
            visual_set, target_set = set(visual_ids), set(target_ids)
            core_set, calib_set = set(core_ids), set(calib_ids)
            if visual_set & target_set or visual_set & core_set or target_set & core_set:
                errors.append("split_manifest audit populations must be mutually disjoint")
            if visual_set | target_set | core_set | calib_set != set(range(total)):
                errors.append("split_manifest partitions must cover every train sample exactly once")
    certificate = json.loads((output / "factor_certificate.json").read_text(encoding="utf-8"))
    edge_admission = json.loads((output / "edge_admission.json").read_text(encoding="utf-8"))
    pilot_run = manifest.get("pilot") is True
    expected_certificate_source = "audit_visual"
    expected_edge_source = "audit_target"
    certificate_pending = certificate.get("status") == "pending"
    edge_pending = edge_admission.get("status") == "pending"
    if (
        (not (v5_run and pilot_run and certificate_pending) and certificate.get("source_split") != expected_certificate_source)
        or (not (v5_run and pilot_run and edge_pending) and edge_admission.get("source_split") != expected_edge_source)
    ):
        errors.append("certificate/edge admission provenance does not match the run version")
    for name, entry in (edge_admission.get("entries") or {}).items():
        if not isinstance(entry, dict) or entry.get("accepted") is not True:
            continue
        metrics = entry.get("metrics")
        if (
            not isinstance(metrics, dict)
            or float(metrics.get("tes_identity_lcb95", 0.0)) <= 0.0
            or float(metrics.get("tes_spatial_lcb95", 0.0)) <= 0.0
        ):
            errors.append(f"accepted edge {name} lacks positive identity/spatial intervention LCBs")
    schedule_rows = _read_jsonl(output / "adaptive_schedule.jsonl")
    schedule_by_epoch = {
        int(row["epoch"]): row for row in schedule_rows
        if isinstance(row.get("epoch"), int)
    }
    for epoch in epochs:
        epoch_dir = output / f"epoch_{epoch:03d}"
        metrics = json.loads((epoch_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        if not {"raw", "deploy_fixed", "test_oracle_diagnostic"} <= set(metrics):
            errors.append(f"epoch_{epoch:03d} metrics lacks raw/deploy/oracle branches")
        if int(metrics.get("sample_count", 0)) <= 0:
            errors.append(f"epoch_{epoch:03d} metrics has no evaluated samples")
        branch = json.loads((epoch_dir / "branch_metrics.json").read_text(encoding="utf-8"))
        if branch.get("available") is not True:
            errors.append(f"epoch_{epoch:03d} branch metrics are unavailable")
        mechanism = json.loads((epoch_dir / "mechanism_summary.json").read_text(encoding="utf-8"))
        if v5_run and not _mechanism_summary_valid(mechanism):
            errors.append(f"epoch_{epoch:03d} mechanism summary is incomplete or changes CREDO semantics")
        calibration_rows = _read_jsonl(epoch_dir / "calibration_stats.jsonl")
        if any(row.get("source_split") != "train_calib" for row in calibration_rows):
            errors.append(f"epoch_{epoch:03d} calibration uses a non-train_calib source")
        reason_rows = _read_jsonl(epoch_dir / "reason_dual_observation_stats.jsonl")
        if v5_run:
            test_reason_rows = [
                row for row in reason_rows
                if row.get("split") == "test" and isinstance(row.get("reason_id"), int)
            ]
            reason_fields = {
                "residual_alpha_mean", "escape_weight_mean", "allowed_factor_mass_mean",
                "disallowed_factor_mass_mean", "reason_factor_mask_area_mean",
                "reason_factor_mask_entropy", "semantic_compatibility_mean",
                "absence_factor_mass_mean", "absence_negative_evidence_mean",
            }
            if (
                len(test_reason_rows) != 21
                or any(not reason_fields <= set(row) for row in test_reason_rows)
                or any(
                    not all(
                        isinstance(row.get(field), (int, float))
                        and torch.isfinite(torch.tensor(float(row[field])))
                        for field in reason_fields
                    )
                    or not 0.0 <= float(row["residual_alpha_mean"]) <= 0.25
                    or not 0.0 <= float(row["escape_weight_mean"]) <= 1.0
                    or not 0.0 <= float(row["allowed_factor_mass_mean"]) <= 1.0
                    or not 0.0 <= float(row["disallowed_factor_mass_mean"]) <= 1e-6
                    for row in test_reason_rows
                )
            ):
                errors.append(f"epoch_{epoch:03d} reason route diagnostics are incomplete or violate ownership")
        hidden_rows = [row for row in reason_rows if row.get("audit") == "hidden_recovery"]
        hidden_grid = {
            (row.get("mode"), float(row.get("hide_fraction", -1.0)))
            for row in hidden_rows
            if row.get("source_split") == "audit_target" and row.get("evaluation_only") is True
        }
        expected_hidden_grid = {
            (mode, fraction)
            for mode in ("mcar", "mar", "mnar")
            for fraction in (0.10, 0.30, 0.50)
        }
        if hidden_grid != expected_hidden_grid or len(hidden_rows) != len(expected_hidden_grid):
            errors.append(f"epoch_{epoch:03d} hidden recovery lacks the leakage-free 10/30/50 audit grid")
        transfer = json.loads((epoch_dir / "target_transfer_summary.json").read_text(encoding="utf-8"))
        transfer_rows = _read_jsonl(epoch_dir / "target_transfer_stats.jsonl")
        transfer_fields = {"factor_id", "target_id", "tet", "tes", "cca", "ap_delta"}
        state_before = schedule_by_epoch.get(epoch, {}).get("state_before")
        if state_before not in {"JOINT_SHADOW", "ADMISSION_CONSOLIDATION"}:
            errors.append(f"epoch_{epoch:03d} has an invalid V5 schedule state")
        # V5 runs an online target probe from epoch zero. A full audit is an
        # additional cadence, never a reason to emit a fake unavailable row.
        if (
            transfer.get("available") is not True
            or transfer.get("source_split") != "audit_target"
            or transfer.get("schema_version") != "mosaic_target_transfer.v2"
            or int(transfer.get("pair_count", transfer.get("target_count", 0))) <= 0
            or transfer.get("audit_level") not in {"online", "full"}
        ):
            errors.append(f"epoch_{epoch:03d} V5 online target transfer is unavailable or lacks audit provenance")
        available_transfer_rows = [row for row in transfer_rows if row.get("available") is True]
        unavailable_transfer_rows = [row for row in transfer_rows if row.get("available") is not True]
        if (
            not transfer_rows
            or not available_transfer_rows
            or any(
                row.get("source_split") != "audit_target"
                or row.get("audit_level") not in {"online", "full"}
                or not transfer_fields <= set(row)
                or row.get("tes_identity") is None
                or row.get("tes_spatial") is None
                or not _matched_control_provenance_valid(row.get("matched_control_arms"))
                for row in available_transfer_rows
            )
            # Candidate edges without an honest same-type/same-region control
            # must remain explicit abstentions, not fake zero-effect rows.
            or any(
                row.get("source_split") != "audit_target"
                or row.get("audit_level") not in {"online", "full"}
                or not isinstance(row.get("unavailable_reason"), str)
                or not row.get("unavailable_reason")
                for row in unavailable_transfer_rows
            )
        ):
            errors.append(f"epoch_{epoch:03d} V5 target transfer rows are incomplete")
        visual_credibility = json.loads((epoch_dir / "visual_credibility.json").read_text(encoding="utf-8"))
        if v5_run and (
            visual_credibility.get("source_split") != "audit_visual"
            or not isinstance(visual_credibility.get("credibility"), list)
        ):
            errors.append(f"epoch_{epoch:03d} visual credibility lacks audit_visual provenance")
        factor_audit = json.loads((epoch_dir / "factor_audit.json").read_text(encoding="utf-8"))
        if v5_run:
            factor_stats = factor_audit.get("factor_stats")
            required_factor_sections = {"counts", "scores", "prototype", "bootstrap_lcb95"}
            if (
                factor_audit.get("source_split") != "audit_visual"
                or not isinstance(factor_stats, dict)
                or not factor_stats
                or any(
                    not isinstance(stats, dict) or not required_factor_sections <= set(stats)
                    for stats in factor_stats.values()
                )
            ):
                errors.append(f"epoch_{epoch:03d} factor audit lacks source-count and content/prior evidence")
        semantic = json.loads((epoch_dir / "semantic_compatibility.json").read_text(encoding="utf-8"))
        utility = json.loads((epoch_dir / "target_utility.json").read_text(encoding="utf-8"))
        if v5_run:
            if (
                semantic.get("source_split") != "audit_target"
                or utility.get("source_split") != "audit_target"
                or semantic.get("available") is not True
                or utility.get("available") is not True
                or semantic.get("audit_level") not in {"online", "full"}
                or utility.get("audit_level") not in {"online", "full"}
                or not isinstance(semantic.get("semantic_compatibility"), list)
                or not isinstance(utility.get("action_target_utility"), list)
            ):
                errors.append(f"epoch_{epoch:03d} target utility artifacts are incomplete or have invalid provenance")
        visual = json.loads((epoch_dir / "visual_audit_manifest.json").read_text(encoding="utf-8"))
        samples = visual.get("samples")
        if (
            visual.get("source_split") != "audit_visual"
            or visual.get("matched_random_control") != "same_factor_equal_mass_spatial_roll"
            or int(visual.get("sample_count", 0)) <= 0
            or not isinstance(samples, list)
            or len(samples) != int(visual.get("sample_count", 0))
        ):
            errors.append(f"epoch_{epoch:03d} visual matched-random audit is invalid")
        elif isinstance(samples, list):
            for sample in samples:
                original_files = sample.get("factor_mask_files")
                random_files = sample.get("matched_random_factor_mask_files")
                if (
                    not isinstance(original_files, list)
                    or not isinstance(random_files, list)
                    or len(original_files) == 0
                    or len(original_files) != len(random_files)
                ):
                    errors.append(f"epoch_{epoch:03d} visual mask file lists are invalid")
                    break
                for original_name, random_name in zip(original_files, random_files):
                    original_path, random_path = epoch_dir / original_name, epoch_dir / random_name
                    if not original_path.is_file() or not random_path.is_file():
                        errors.append(f"epoch_{epoch:03d} visual mask entity is missing")
                        break
                    original_mask = torch.load(original_path, map_location="cpu", weights_only=True)
                    random_mask = torch.load(random_path, map_location="cpu", weights_only=True)
                    expected = torch.roll(
                        original_mask,
                        shifts=(original_mask.shape[-2] // 3, original_mask.shape[-1] // 3),
                        dims=(-2, -1),
                    )
                    if not torch.equal(random_mask, expected) or not torch.isclose(original_mask.sum(), random_mask.sum()):
                        errors.append(f"epoch_{epoch:03d} visual matched-random mask is not the same-factor equal-mass roll")
                        break
        gradient_rows = _read_jsonl(epoch_dir / "gradient_ownership.jsonl")
        required_gradient_fields = {"epoch", "step", "loss", "owner_group", "grad_norm", "finite"}
        if (
            not gradient_rows
            or any(not required_gradient_fields <= set(row) or row.get("finite") is not True for row in gradient_rows)
            or (v5_run and not _gradient_firewall_rows_valid(gradient_rows))
        ):
            errors.append(f"epoch_{epoch:03d} gradient ownership audit is incomplete")
        names = json.loads((epoch_dir / "logits" / "file_names.json").read_text(encoding="utf-8"))
        if not isinstance(names, list) or not names:
            errors.append(f"epoch_{epoch:03d} has no test file names")
        for name in ICDOR_LOGIT_FILES:
            tensor = torch.load(epoch_dir / "logits" / name, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or not tensor.numel() or not torch.isfinite(tensor).all():
                errors.append(f"epoch_{epoch:03d}/{name} is empty or non-finite")
            elif tensor.shape[0] != len(names):
                errors.append(f"epoch_{epoch:03d}/{name} does not align with file_names")

        # V5 diagnostics are semantic contracts, not merely non-empty files.
        credibility_rows = _read_jsonl(epoch_dir / "credibility_stats.jsonl")
        if v5_run and (not credibility_rows or any(
            row.get("split") != "test"
            or not isinstance(row.get("factor_id"), int)
            or any(
                not isinstance(row.get(field), (int, float))
                or not torch.isfinite(torch.tensor(float(row[field])))
                or not 0.0 <= float(row[field]) <= 1.0
                for field in (
                    "cV_mean", "cV_p50", "cV_p95", "cV_ema_mean",
                    "cV_nonzero_rate", "cV_route_effective_mean",
                )
            )
            for row in credibility_rows
        )):
            errors.append(f"epoch_{epoch:03d} credibility_stats lacks finite bounded cV rows")
        fine_rows = _read_jsonl(epoch_dir / "fine_transport_stats.jsonl")
        if v5_run and (not fine_rows or any(
            row.get("split") != "test"
            or row.get("typed_coordinates_present") is not True
            or not all(
                isinstance(row.get(field), (int, float))
                and torch.isfinite(torch.tensor(float(row[field])))
                for field in (
                    "fine_mask_delta_mean", "fine_mask_delta_max", "anchor_separation_mean",
                    "fine_off_action_shadow_delta_abs_mean", "fine_off_reason_latent_delta_abs_mean",
                    "coarse_off_action_shadow_delta_abs_mean", "coarse_off_reason_latent_delta_abs_mean",
                )
            )
            for row in fine_rows
        ) or not any(float(row.get("fine_mask_delta_mean", 0.0)) > 1e-8 for row in fine_rows)):
            errors.append(f"epoch_{epoch:03d} fine_transport_stats does not prove typed fine evidence differs from coarse")
        route_rows = _read_jsonl(epoch_dir / "route_ownership.jsonl")
        if v5_run and (not route_rows or not any(row.get("summary") == "per_action_route_effect" for row in route_rows)):
            errors.append(f"epoch_{epoch:03d} route_ownership lacks per-action ownership diagnostics")
        if v5_run:
            for row in route_rows:
                if row.get("route_mode") == "shadow" and row.get("action_final_visual_equal") is not True:
                    errors.append(f"epoch_{epoch:03d} shadow route changed final action before admission")
                    break
    result["semantic_errors"] = errors
    result["pass"] = result["pass"] and not errors
    return result
