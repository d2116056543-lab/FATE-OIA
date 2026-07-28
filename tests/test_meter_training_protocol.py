from pathlib import Path

import yaml


def test_training_protocol_is_test_only_and_has_disjoint_calibration() -> None:
    config = yaml.safe_load(Path("configs/fate_oia_train_360x640_acpr_meter_oia_v1.yaml").read_text(encoding="utf-8"))
    assert config["runtime"]["test_only"] is True
    assert config["posthoc_calibration"]["fit_split"] == "train_calib"
    assert config["best_selection_split"] == "test"
    assert config["splits"]["main_audit_calib_disjoint"] is True
