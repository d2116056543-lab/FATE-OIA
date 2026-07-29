import inspect
import math

from fate_oia.engine import train_acpr_meter_oia as trainer
from fate_oia.engine import evaluate_tesa_pilot as pilot_evaluator
from fate_oia.engine.evaluate_tesa_pilot import (
    evaluate_admission_with_continuous_action_identity,
)
from fate_oia.utils.tesa_contracts import (
    TWO_EPOCH_ADMISSION_RULES,
    build_runtime_subset_counts,
    evaluate_two_epoch_admission,
    factor_groundability_tiers,
    validate_runtime_subset_counts,
)


def test_runtime_subset_builder_writes_one_canonical_schema() -> None:
    counts = build_runtime_subset_counts(
        {"main": range(4096), "audit": range(1024), "calib": range(512)},
        test_count=512,
    )

    assert counts == {
        "train_main": 4096,
        "train_audit": 1024,
        "train_calib": 512,
        "test": 512,
    }
    assert validate_runtime_subset_counts(counts, counts) == []
    assert validate_runtime_subset_counts(
        {"main": 4096, "audit": 1024, "calib": 512}, counts
    ) == ["train_main", "train_audit", "train_calib", "test"]
    source = inspect.getsource(trainer.train)
    assert "build_runtime_subset_counts(" in source
    assert "test_count=len(test_indices)" in source


def test_pilot_reuse_allows_only_offline_evaluation_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        pilot_evaluator.subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "fate_oia/engine/evaluate_tesa_pilot.py\n"
            "tests/test_tesa_protocol_artifact_contract.py\n"
        ),
    )
    assert pilot_evaluator._evaluation_only_git_delta("pilot", "current")

    monkeypatch.setattr(
        pilot_evaluator.subprocess,
        "check_output",
        lambda *args, **kwargs: "fate_oia/models/meter_oia_model.py\n",
    )
    assert not pilot_evaluator._evaluation_only_git_delta("pilot", "current")


def test_state_admission_counts_multiclass_confusion_rows() -> None:
    rows = pilot_evaluator._state_rows_for_admission(
        [
            {
                "factor_id": 0,
                "source_count": 751,
                "state_auprc": 0.85,
                "state_frequency_baseline": 0.76,
                "state_confusion_matrix": [
                    [569, 6, 0],
                    [169, 7, 0],
                    [0, 0, 0],
                ],
            },
            {
                "factor_id": 1,
                "source_count": 0,
                "state_auprc": None,
                "state_frequency_baseline": None,
                "state_confusion_matrix": [[0, 0, 0]] * 3,
            },
        ],
        {"factor_weight_mean_by_action_factor": [[1.0, 0.0]]},
    )
    assert rows[0]["positive_count"] == 575
    assert rows[0]["negative_count"] == 176
    assert math.isnan(rows[1]["prevalence"])


def test_schema_tiers_and_prevalence_one_rows_do_not_create_impossible_gate() -> None:
    tiers = factor_groundability_tiers(
        [
            {"id": 0, "groundability": "full"},
            {"id": 1, "groundability": "partial"},
            {"id": 2, "groundability": "latent"},
        ]
    )
    assert tiers == {"full": (0,), "partial": (1,), "latent": (2,)}

    state_rows = [
        {
            "factor_id": factor_id,
            "prevalence": 0.45,
            "auprc": 0.50,
            "positive_count": 24,
            "negative_count": 30,
            "observed_usage_share": 0.10,
            "source_eligible_opportunity_share": 0.10,
        }
        for factor_id in range(6)
    ]
    state_rows.append(
        {
            "factor_id": 7,
            "prevalence": 1.0,
            "auprc": 1.0,
            "positive_count": 120,
            "negative_count": 0,
            "observed_usage_share": 0.0,
            "source_eligible_opportunity_share": 0.0,
        }
    )
    decision = evaluate_two_epoch_admission(
        {
            "protocol_ok": True,
            "artifact_ok": True,
            "numerics_ok": True,
            "implementation_audit_ok": True,
            "gradient_ownership_ok": True,
            "unknown_mask_ok": True,
            "source_completeness_ok": True,
            "no_test_leakage_ok": True,
            "paired_epoch_count": 2,
            "null_semantics_ok": True,
            "eligible_factor_coverage": list(range(12)),
            "executed_factor_coverage": list(range(12)),
            "action_correction_rms_ratio": [0.02, 0.05, 0.10, 0.20],
            "action_map_deltas": [0.004, 0.002],
            "transport_target_effect": [0.01, 0.02, 0.03, 0.0005],
            "reason_ap_delta": 0.001,
            "reason_f1_delta": -0.001,
            "reason_ap_delta_ci": {
                "mean": 0.001,
                "low": 0.0001,
                "high": 0.002,
                "sample_count": 512,
            },
            "deletion_gap_ci": {
                "mean": 0.02,
                "low": 0.004,
                "high": 0.03,
                "cluster_count": 128,
            },
            "state_rows": state_rows,
        }
    )

    assert TWO_EPOCH_ADMISSION_RULES["min_eligible_state_factors"] == 6
    assert TWO_EPOCH_ADMISSION_RULES["min_reason_ap_delta"] == 0.001
    assert decision["pass"]
    assert decision["truth_table"]["evidence_state"]["eligible_rows"] == list(range(6))
    assert decision["truth_table"]["evidence_state"]["excluded_prevalence_one"] == [7]
    assert decision["truth_table"]["evidence_state"]["factor2_usage_excess"] == 0.0


def test_two_epoch_admission_marks_ci_crossing_zero_as_inconclusive_not_negative() -> None:
    decision = evaluate_two_epoch_admission(
        {
            "protocol_ok": True,
            "artifact_ok": True,
            "numerics_ok": True,
            "implementation_audit_ok": True,
            "gradient_ownership_ok": True,
            "unknown_mask_ok": True,
            "source_completeness_ok": True,
            "no_test_leakage_ok": True,
            "paired_epoch_count": 2,
            "null_semantics_ok": True,
            "eligible_factor_coverage": list(range(12)),
            "executed_factor_coverage": list(range(12)),
            "action_correction_rms_ratio": [0.02, 0.05, 0.10, 0.20],
            "action_map_deltas": [0.004, 0.002],
            "transport_target_effect": [0.01, 0.02, 0.03, 0.04],
            "reason_ap_delta": 0.001,
            "reason_f1_delta": 0.0,
            "reason_ap_delta_ci": {
                "mean": 0.001,
                "low": 0.0001,
                "high": 0.002,
                "sample_count": 512,
            },
            "deletion_gap_ci": {
                "mean": 0.02,
                "low": -0.001,
                "high": 0.04,
                "cluster_count": 128,
            },
            "state_rows": [
                {
                    "factor_id": factor_id,
                    "prevalence": 0.4,
                    "auprc": 0.5,
                    "positive_count": 20,
                    "negative_count": 20,
                    "observed_usage_share": 0.1,
                    "source_eligible_opportunity_share": 0.1,
                }
                for factor_id in range(6)
            ],
        }
    )

    assert decision["pass"]
    assert decision["truth_table"]["deletion"]["status"] == "inconclusive"
    assert not decision["truth_table"]["deletion"]["pass"]


def test_two_epoch_admission_requires_action_and_two_of_three_mechanism_classes() -> None:
    base = {
        "protocol_ok": True,
        "artifact_ok": True,
        "numerics_ok": True,
        "implementation_audit_ok": True,
        "gradient_ownership_ok": True,
        "unknown_mask_ok": True,
        "source_completeness_ok": True,
        "no_test_leakage_ok": True,
        "paired_epoch_count": 2,
        "null_semantics_ok": False,
        "eligible_factor_coverage": [],
        "executed_factor_coverage": [],
        "action_correction_rms_ratio": [0.02, 0.05, 0.10, 0.20],
        "action_map_deltas": [0.003, 0.001],
        "transport_target_effect": [0.002, 0.002, 0.002, 0.0],
        "reason_ap_delta": 0.001,
        "reason_f1_delta": 0.0,
        "reason_ap_delta_ci": {
            "mean": 0.001,
            "low": 0.0001,
            "high": 0.002,
            "sample_count": 512,
        },
        "deletion_gap_ci": {
            "mean": 0.02,
            "low": 0.001,
            "high": 0.04,
            "cluster_count": 128,
        },
        "state_rows": [],
    }
    decision = evaluate_two_epoch_admission(base)

    assert decision["truth_table"]["action"]["pass"]
    assert not decision["truth_table"]["evidence_state"]["pass"]
    assert decision["truth_table"]["reason"]["pass"]
    assert decision["truth_table"]["deletion"]["pass"]
    assert decision["pass"]


def test_action_admission_accepts_continuous_delta_identity_when_ap_is_tied() -> None:
    metrics = {
        "protocol_ok": True,
        "artifact_ok": True,
        "numerics_ok": True,
        "implementation_audit_ok": True,
        "gradient_ownership_ok": True,
        "unknown_mask_ok": True,
        "source_completeness_ok": True,
        "no_test_leakage_ok": True,
        "paired_epoch_count": 2,
        "null_semantics_ok": False,
        "action_correction_rms_ratio": [0.04, 0.05, 0.03, 0.09],
        "action_map_deltas": [0.0004, 0.0001],
        "transport_target_effect": [0.0, 0.0, 0.0, 0.0],
        "action_delta_auc": [0.80, 0.66, 0.61, 0.58],
        "action_delta_separation": [0.02, 0.004, 0.004, 0.003],
        "reason_ap_delta": 0.002,
        "reason_f1_delta": 0.0,
        "reason_ap_delta_ci": {
            "mean": 0.002,
            "low": 0.001,
            "high": 0.003,
            "sample_count": 512,
        },
        "deletion_gap_ci": {
            "mean": 0.01,
            "low": 0.004,
            "high": 0.02,
            "cluster_count": 128,
        },
        "state_rows": [],
    }
    decision = evaluate_admission_with_continuous_action_identity(metrics)
    action = decision["truth_table"]["action"]
    assert action["continuous_identity_pass"]
    assert not action["discrete_identity_pass"]
    assert action["pass"]
    assert decision["pass"]


def test_deletion_and_reason_bootstrap_cannot_pass_on_tiny_sample() -> None:
    decision = evaluate_two_epoch_admission(
        {
            "protocol_ok": True,
            "artifact_ok": True,
            "numerics_ok": True,
            "implementation_audit_ok": True,
            "gradient_ownership_ok": True,
            "unknown_mask_ok": True,
            "source_completeness_ok": True,
            "no_test_leakage_ok": True,
            "paired_epoch_count": 2,
            "null_semantics_ok": False,
            "action_correction_rms_ratio": [0.02, 0.05, 0.10, 0.20],
            "action_map_deltas": [0.003, 0.001],
            "transport_target_effect": [0.002, 0.002, 0.002, 0.0],
            "reason_ap_delta": 0.01,
            "reason_f1_delta": 0.0,
            "reason_ap_delta_ci": {
                "mean": 0.01,
                "low": 0.009,
                "high": 0.011,
                "sample_count": 1,
            },
            "deletion_gap_ci": {
                "mean": 0.02,
                "low": 0.019,
                "high": 0.021,
                "cluster_count": 1,
            },
            "state_rows": [],
        }
    )
    assert not decision["truth_table"]["reason"]["pass"]
    assert decision["truth_table"]["deletion"]["status"] == "insufficient_samples"
    assert not decision["truth_table"]["deletion"]["pass"]
    assert not decision["pass"]
