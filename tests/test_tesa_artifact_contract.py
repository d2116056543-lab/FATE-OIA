import json

import torch

from fate_oia.engine.tesa_diagnostics import REQUIRED_TESA_ARTIFACT_FIELDS
from fate_oia.utils.meter_artifacts import validate_epoch_artifacts


def test_artifact_contract_contains_mechanism_and_calibration_fields() -> None:
    required = {
        "action_factor_contributions", "state_confusion_matrix",
        "unique_sample_count", "temperature", "threshold_vector",
        "train_calib_raw_joint", "train_calib_deploy_joint", "fallback_reason",
        "dino_time", "reserved_gb",
    }
    assert required <= REQUIRED_TESA_ARTIFACT_FIELDS


def test_artifact_validator_checks_shapes_alignment_and_finite(tmp_path) -> None:
    for name in ("metrics_raw.json", "metrics_deploy.json", "branch_metrics.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    typed = {
        "state_confusion_matrix": [[[0] * 3 for _ in range(3)] for _ in range(21)],
        "source_coverage": [0] * 21,
        "same_type_margin": [None] * 21,
        "mirror_equivariance": [None] * 21,
        "identity_target_delta": [0.0] * 4,
        "identity_wrong_delta": [0.0] * 4,
        "identity_ap_delta_matrix": [[0.0] * 4 for _ in range(4)],
        "factor_off_delta": [0.0] * 4,
        "state_off_delta": [0.0] * 4,
        "cross_sample_swap_effect": [0.0] * 4,
        "train_audit": {"per_factor": [{} for _ in range(21)]},
        "patch_audit": {
            "unique_sample_count": 2,
            "action_coverage": [0, 1, 2, 3],
            "factor_coverage": list(range(12)),
            "eligible_factor_coverage": list(range(12)),
            "requested_factor_coverage": list(range(12)),
            "executed_factor_coverage": list(range(12)),
            "model_top_factor_coverage": list(range(12)),
            "selected_minus_control_ci": {
                "mean": 0.01,
                "low": 0.001,
                "high": 0.02,
                "n_bootstrap": 100,
                "cluster_count": 2,
            },
        },
    }
    (tmp_path / "typed_evidence.json").write_text(
        json.dumps(typed), encoding="utf-8"
    )
    (tmp_path / "pu_stats.json").write_text(
        json.dumps({"active_labels": [], "lambda": [0.0] * 21, "labels": []}),
        encoding="utf-8",
    )
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "theta": [0.0] * 25,
                "temperature": None,
                "strategy": "global",
                "accepted": True,
                "fallback_reason": "",
                "fit_split": "train_calib",
                "representation_updated": False,
                "train_calib_raw_joint": 0.0,
                "train_calib_deploy_joint": 0.0,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime.json").write_text(
        json.dumps(
            {
                "epoch": 0,
                "train_rows": 1,
                "mean_data_time": 0.1,
                "mean_dino_time": 0.1,
                "peak_reserved_gb": 1.0,
                "eval_mode_time": {},
                "dino_call_count": {"main": 1},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "file_names_test.json").write_text(
        json.dumps({"file_names": ["a.jpg", "b.jpg"]}), encoding="utf-8"
    )
    tensors = {
        "logits_action_final_raw_test.pt": torch.zeros(2, 4),
        "logits_reason_final_raw_test.pt": torch.zeros(2, 21),
        "logits_action_visual_test.pt": torch.zeros(2, 4),
        "logits_reason_global_test.pt": torch.zeros(2, 21),
        "labels_action_test.pt": torch.zeros(2, 4),
        "labels_reason_test.pt": torch.zeros(2, 21),
    }
    for name, value in tensors.items():
        torch.save(value, tmp_path / name)
    assert validate_epoch_artifacts(tmp_path) == []

    torch.save(torch.full((2, 4), float("nan")), tmp_path / "logits_action_final_raw_test.pt")
    failures = validate_epoch_artifacts(tmp_path)
    assert "logits_action_final_raw_test.pt:non_finite" in failures

    torch.save(torch.zeros(1, 21), tmp_path / "labels_reason_test.pt")
    failures = validate_epoch_artifacts(tmp_path)
    assert "labels_reason_test.pt:shape" in failures


def test_artifact_validator_rejects_incomplete_tesa_mechanism_schema(tmp_path) -> None:
    test_artifact_validator_checks_shapes_alignment_and_finite(tmp_path)
    torch.save(torch.zeros(2, 4), tmp_path / "logits_action_final_raw_test.pt")
    torch.save(torch.zeros(2, 21), tmp_path / "labels_reason_test.pt")
    (tmp_path / "typed_evidence.json").write_text("{}", encoding="utf-8")

    failures = validate_epoch_artifacts(tmp_path)
    assert "typed_evidence.json:mechanism_schema" in failures


def test_artifact_validator_rejects_nested_shape_and_nan(tmp_path) -> None:
    test_artifact_validator_checks_shapes_alignment_and_finite(tmp_path)
    torch.save(torch.zeros(2, 4), tmp_path / "logits_action_final_raw_test.pt")
    torch.save(torch.zeros(2, 21), tmp_path / "labels_reason_test.pt")
    typed_path = tmp_path / "typed_evidence.json"
    typed = json.loads(typed_path.read_text(encoding="utf-8"))
    typed["state_confusion_matrix"][0] = [[0, 0]]
    typed["identity_target_delta"][0] = float("nan")
    typed_path.write_text(json.dumps(typed), encoding="utf-8")

    failures = validate_epoch_artifacts(tmp_path)
    assert "typed_evidence.json:mechanism_schema" in failures


def test_artifact_validator_rejects_inconsistent_identity_derivatives(tmp_path) -> None:
    test_artifact_validator_checks_shapes_alignment_and_finite(tmp_path)
    torch.save(torch.zeros(2, 4), tmp_path / "logits_action_final_raw_test.pt")
    torch.save(torch.zeros(2, 21), tmp_path / "labels_reason_test.pt")
    typed_path = tmp_path / "typed_evidence.json"
    typed = json.loads(typed_path.read_text(encoding="utf-8"))
    typed["identity_ap_delta_matrix"][0][0] = 0.5
    typed_path.write_text(json.dumps(typed), encoding="utf-8")

    assert (
        "typed_evidence.json:mechanism_schema"
        in validate_epoch_artifacts(tmp_path)
    )
