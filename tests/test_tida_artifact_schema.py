from fate_oia.utils.tida_artifacts import validate_completion_artifact
from pathlib import Path


def test_completion_artifact_fails_closed_on_missing_binding():
    failures = validate_completion_artifact({"pass": True}, phase="full_train_ready")
    assert failures and "git_head" in failures
    assert "golden_oracle_sha256" in failures


def test_epoch_export_contains_formal_temporal_chain():
    source = Path("fate_oia/engine/evaluate_tida_oia.py").read_text(encoding="utf-8")
    for key in (
        "terminal_prediction_history", "terminal_target_evidence", "innovation_token",
        "predicate_region_mass_velocity", "action_temporal_route", "action_factor_contribution",
        "reason_temporal_route", "frame_valid_mask", "timestamps",
    ):
        assert f'"{key}"' in source
