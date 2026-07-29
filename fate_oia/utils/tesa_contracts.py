"""Shared, data-only contracts for TESA pilot artifacts and admission checks."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


CANONICAL_RUNTIME_SPLITS = (
    "train_main",
    "train_audit",
    "train_calib",
    "test",
)
RUNTIME_SPLIT_ALIASES = {
    "main": "train_main",
    "audit": "train_audit",
    "calib": "train_calib",
}
PATCH_AUDIT_COVERAGE_FIELDS = (
    "eligible_factor_coverage",
    "requested_factor_coverage",
    "executed_factor_coverage",
    "model_top_factor_coverage",
)

# Approved two-epoch diagnostic criteria; these test mechanism health only.
TWO_EPOCH_ADMISSION_RULES = {
    "min_state_positive_count": 20,
    "min_state_negative_count": 20,
    "min_eligible_state_factors": 6,
    "min_state_macro_auprc_excess": 0.02,
    "min_state_above_prevalence_fraction": 2.0 / 3.0,
    "max_factor2_usage_excess": 0.15,
    "min_action_correction_rms_ratio": 0.02,
    "max_action_correction_rms_ratio": 0.20,
    "min_action_mean_map_delta": 0.0,
    "min_action_epoch_map_delta": -0.002,
    "min_positive_action_identity_effects": 3,
    "min_action_delta_auc": 0.55,
    "min_positive_action_delta_separations": 4,
    "min_action_identity_effect": 0.001,
    "min_reason_ap_delta": 0.001,
    "min_reason_f1_delta": -0.003,
    "min_deletion_gap_mean": 0.0,
    "min_deletion_cluster_count": 128,
    "min_reason_bootstrap_sample_count": 128,
    "min_mechanism_classes": 2,
}


def _count(value: Any) -> int:
    if isinstance(value, (str, bytes)):
        raise TypeError("runtime subset counts cannot be strings")
    return value if isinstance(value, int) else len(value)


def build_runtime_subset_counts(
    train_splits: Mapping[str, Any], *, test_count: int | Sequence[Any]
) -> dict[str, int]:
    """Build the one canonical split schema for writer and evaluator."""
    result: dict[str, int] = {}
    for source, canonical in RUNTIME_SPLIT_ALIASES.items():
        if source not in train_splits:
            raise KeyError(f"Missing runtime split: {source}")
        result[canonical] = _count(train_splits[source])
    result["test"] = _count(test_count)
    return result


def validate_runtime_subset_counts(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    """Return canonical split names whose counts are absent or mismatched."""
    failures: list[str] = []
    for name in CANONICAL_RUNTIME_SPLITS:
        try:
            valid = int(actual.get(name, -1)) == int(expected[name])
        except (TypeError, ValueError):
            valid = False
        if not valid:
            failures.append(name)
    return failures


def factor_groundability_tiers(
    factors: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, ...]]:
    """Partition schema factors without treating partial or latent as full."""
    tiers: dict[str, list[int]] = {"full": [], "partial": [], "latent": []}
    for factor in factors:
        tier = str(factor.get("groundability", "latent"))
        if tier not in tiers:
            raise ValueError(f"Unsupported groundability tier: {tier}")
        tiers[tier].append(int(factor["id"]))
    return {name: tuple(sorted(ids)) for name, ids in tiers.items()}


def patch_audit_contract_failures(patch: Mapping[str, Any]) -> list[str]:
    """Validate coverage and CI fields without conflating their semantics."""
    failures: list[str] = []
    for name in PATCH_AUDIT_COVERAGE_FIELDS:
        values = patch.get(name)
        if not (
            isinstance(values, list)
            and all(isinstance(value, int) and 0 <= value < 21 for value in values)
        ):
            failures.append(name)
    ci = patch.get("selected_minus_control_ci")
    if not (
        isinstance(ci, Mapping)
        and all(
            isinstance(ci.get(name), (int, float))
            and math.isfinite(float(ci[name]))
            for name in ("mean", "low", "high")
        )
        and float(ci["low"]) <= float(ci["mean"]) <= float(ci["high"])
        and int(ci.get("n_bootstrap", 0)) >= 1
        and int(ci.get("cluster_count", 0)) >= 1
    ):
        failures.append("selected_minus_control_ci")
    return failures


def evaluate_two_epoch_admission(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the approved two-epoch diagnostic truth table."""
    state_rows = [row for row in metrics.get("state_rows", []) if isinstance(row, Mapping)]
    eligible_rows = [
        row
        for row in state_rows
        if 0.0 < float(row.get("prevalence", 0.0)) < 1.0
        and isinstance(row.get("auprc"), (int, float))
        and math.isfinite(float(row["auprc"]))
        and int(row.get("positive_count", 0))
        >= TWO_EPOCH_ADMISSION_RULES["min_state_positive_count"]
        and int(row.get("negative_count", 0))
        >= TWO_EPOCH_ADMISSION_RULES["min_state_negative_count"]
    ]
    excluded_prevalence_one = [
        int(row.get("factor_id", -1))
        for row in state_rows
        if float(row.get("prevalence", 0.0)) >= 1.0
    ]
    transport = [float(value) for value in metrics.get("transport_target_effect", [])]
    delta_auc = [float(value) for value in metrics.get("action_delta_auc", [])]
    delta_separation = [
        float(value) for value in metrics.get("action_delta_separation", [])
    ]
    correction_ratios = [
        float(value) for value in metrics.get("action_correction_rms_ratio", [])
    ]
    action_map_deltas = [
        float(value) for value in metrics.get("action_map_deltas", [])
    ]
    deletion_ci = metrics.get("deletion_gap_ci", {})
    factor2 = next(
        (row for row in eligible_rows if int(row.get("factor_id", -1)) == 2),
        None,
    )
    factor2_usage_excess = (
        float(factor2.get("observed_usage_share", 0.0))
        - float(factor2.get("source_eligible_opportunity_share", 0.0))
        if factor2 is not None
        else float("inf")
    )
    state_excesses = [
        float(row["auprc"]) - float(row["prevalence"])
        for row in eligible_rows
    ]
    state_macro_excess = (
        sum(state_excesses) / len(state_excesses)
        if state_excesses
        else float("nan")
    )
    state_above_prevalence_fraction = (
        sum(value > 0.0 for value in state_excesses) / len(state_excesses)
        if state_excesses
        else 0.0
    )
    deletion_mean = float(deletion_ci.get("mean", float("nan"))) if isinstance(
        deletion_ci, Mapping
    ) else float("nan")
    deletion_low = float(deletion_ci.get("low", float("nan"))) if isinstance(
        deletion_ci, Mapping
    ) else float("nan")
    deletion_high = float(deletion_ci.get("high", float("nan"))) if isinstance(
        deletion_ci, Mapping
    ) else float("nan")
    deletion_cluster_count = int(deletion_ci.get("cluster_count", 0)) if isinstance(
        deletion_ci, Mapping
    ) else 0
    if deletion_cluster_count < TWO_EPOCH_ADMISSION_RULES["min_deletion_cluster_count"]:
        deletion_status = "insufficient_samples"
    elif deletion_mean <= TWO_EPOCH_ADMISSION_RULES["min_deletion_gap_mean"]:
        deletion_status = "negative"
    elif deletion_low <= 0.0 <= deletion_high:
        deletion_status = "inconclusive"
    elif deletion_low > 0.0:
        deletion_status = "positive"
    else:
        deletion_status = "invalid"
    deterministic_pass = (
        bool(metrics.get("protocol_ok"))
        and bool(metrics.get("artifact_ok"))
        and bool(metrics.get("numerics_ok", metrics.get("losses_finite")))
        and bool(metrics.get("implementation_audit_ok"))
        and bool(metrics.get("gradient_ownership_ok"))
        and bool(metrics.get("unknown_mask_ok"))
        and bool(metrics.get("source_completeness_ok"))
        and bool(metrics.get("no_test_leakage_ok"))
        and int(metrics.get("paired_epoch_count", 0)) == 2
    )
    discrete_identity_pass = (
        sum(
            value > TWO_EPOCH_ADMISSION_RULES["min_action_identity_effect"]
            for value in transport
        )
        >= TWO_EPOCH_ADMISSION_RULES["min_positive_action_identity_effects"]
    )
    continuous_identity_pass = (
        len(delta_auc) == 4
        and all(
            value >= TWO_EPOCH_ADMISSION_RULES["min_action_delta_auc"]
            for value in delta_auc
        )
        and sum(value > 0.0 for value in delta_separation)
        >= TWO_EPOCH_ADMISSION_RULES[
            "min_positive_action_delta_separations"
        ]
    )
    action_pass = (
        len(correction_ratios) == 4
        and all(
            TWO_EPOCH_ADMISSION_RULES["min_action_correction_rms_ratio"]
            <= value
            <= TWO_EPOCH_ADMISSION_RULES["max_action_correction_rms_ratio"]
            for value in correction_ratios
        )
        and bool(action_map_deltas)
        and sum(action_map_deltas) / len(action_map_deltas)
        > TWO_EPOCH_ADMISSION_RULES["min_action_mean_map_delta"]
        and all(
            value >= TWO_EPOCH_ADMISSION_RULES["min_action_epoch_map_delta"]
            for value in action_map_deltas
        )
        and (discrete_identity_pass or continuous_identity_pass)
    )
    evidence_state_pass = (
        bool(metrics.get("null_semantics_ok"))
        and len(eligible_rows)
        >= TWO_EPOCH_ADMISSION_RULES["min_eligible_state_factors"]
        and state_macro_excess
        >= TWO_EPOCH_ADMISSION_RULES["min_state_macro_auprc_excess"]
        and state_above_prevalence_fraction
        >= TWO_EPOCH_ADMISSION_RULES["min_state_above_prevalence_fraction"]
        and factor2_usage_excess
        <= TWO_EPOCH_ADMISSION_RULES["max_factor2_usage_excess"]
    )
    reason_ci = metrics.get("reason_ap_delta_ci", {})
    reason_pass = (
        float(metrics.get("reason_ap_delta", float("nan")))
        >= TWO_EPOCH_ADMISSION_RULES["min_reason_ap_delta"]
        and float(metrics.get("reason_f1_delta", float("nan")))
        >= TWO_EPOCH_ADMISSION_RULES["min_reason_f1_delta"]
        and isinstance(reason_ci, Mapping)
        and int(reason_ci.get("sample_count", 0))
        >= TWO_EPOCH_ADMISSION_RULES["min_reason_bootstrap_sample_count"]
        and float(reason_ci.get("low", float("-inf"))) > 0.0
    )
    deletion_pass = deletion_status == "positive"
    truth_table = {
        "deterministic": {
            "protocol_ok": bool(metrics.get("protocol_ok")),
            "artifact_ok": bool(metrics.get("artifact_ok")),
            "numerics_ok": bool(metrics.get("numerics_ok", metrics.get("losses_finite"))),
            "implementation_audit_ok": bool(
                metrics.get("implementation_audit_ok")
            ),
            "gradient_ownership_ok": bool(metrics.get("gradient_ownership_ok")),
            "unknown_mask_ok": bool(metrics.get("unknown_mask_ok")),
            "source_completeness_ok": bool(
                metrics.get("source_completeness_ok")
            ),
            "no_test_leakage_ok": bool(metrics.get("no_test_leakage_ok")),
            "paired_epoch_count": int(metrics.get("paired_epoch_count", 0)),
            "pass": deterministic_pass,
        },
        "action": {
            "correction_rms_ratio": correction_ratios,
            "map_deltas": action_map_deltas,
            "identity_target_effect": transport,
            "delta_auc": delta_auc,
            "delta_separation": delta_separation,
            "discrete_identity_pass": discrete_identity_pass,
            "continuous_identity_pass": continuous_identity_pass,
            "pass": action_pass,
        },
        "evidence_state": {
            "eligible_rows": [int(row.get("factor_id", -1)) for row in eligible_rows],
            "excluded_prevalence_one": excluded_prevalence_one,
            "macro_auprc_excess": state_macro_excess,
            "above_prevalence_fraction": state_above_prevalence_fraction,
            "factor2_usage_excess": factor2_usage_excess,
            "null_semantics_ok": bool(metrics.get("null_semantics_ok")),
            "pass": evidence_state_pass,
        },
        "reason": {
            "ap_delta": float(metrics.get("reason_ap_delta", float("nan"))),
            "f1_delta": float(metrics.get("reason_f1_delta", float("nan"))),
            "ap_delta_ci": dict(reason_ci) if isinstance(reason_ci, Mapping) else {},
            "pass": reason_pass,
        },
        "deletion": {
            "ci": dict(deletion_ci) if isinstance(deletion_ci, Mapping) else {},
            "status": deletion_status,
            "pass": deletion_pass,
        },
    }
    mechanism_classes = ("evidence_state", "reason", "deletion")
    mechanism_pass_count = sum(
        bool(truth_table[name]["pass"]) for name in mechanism_classes
    )
    return {
        "pass": (
            deterministic_pass
            and action_pass
            and mechanism_pass_count
            >= TWO_EPOCH_ADMISSION_RULES["min_mechanism_classes"]
        ),
        "rules": dict(TWO_EPOCH_ADMISSION_RULES),
        "mechanism_pass_count": mechanism_pass_count,
        "truth_table": truth_table,
    }
