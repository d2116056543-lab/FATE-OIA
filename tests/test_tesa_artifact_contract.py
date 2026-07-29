from fate_oia.engine.tesa_diagnostics import REQUIRED_TESA_ARTIFACT_FIELDS


def test_artifact_contract_contains_mechanism_and_calibration_fields() -> None:
    required = {
        "action_factor_contributions", "state_confusion_matrix",
        "unique_sample_count", "temperature", "threshold_vector",
        "train_calib_raw_joint", "train_calib_deploy_joint", "fallback_reason",
        "dino_time", "reserved_gb",
    }
    assert required <= REQUIRED_TESA_ARTIFACT_FIELDS
