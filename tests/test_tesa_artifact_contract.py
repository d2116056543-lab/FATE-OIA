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
    for name in (
        "metrics_raw.json", "metrics_deploy.json", "branch_metrics.json",
        "typed_evidence.json", "pu_stats.json", "calibration.json", "runtime.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
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
