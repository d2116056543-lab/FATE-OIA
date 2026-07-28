from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_meter_config_preserves_direct_image_contract() -> None:
    config_path = ROOT / "configs" / "fate_oia_train_360x640_acpr_meter_oia_v1.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert data["experiment"]["name"] == "acpr_meter_oia_v1"
    assert data["experiment"]["direct_image"] is True
    assert data["runtime"]["test_only"] is True
    assert data["runtime"]["no_feature_cache"] is True
    assert data["runtime"]["require_no_token_compression"] is True
    assert data["model"]["token_compression"] == "none"
    assert data["model"]["feature_cache_enabled"] is False
    assert data["model"]["use_pair_memory"] is False
    assert data["model"]["use_action_set_final"] is False
    assert data["model"]["use_trainable_threshold"] is False
    assert data["model"]["use_trainable_calibration"] is False
    assert data["data"]["image_height"] == 360
    assert data["data"]["image_width"] == 640
    assert data["backbone"]["selected_layers"] == [3, 7, 11]
    assert data["backbone"]["freeze_backbone"] is True


def test_meter_runtime_candidates_and_split_contract_are_explicit() -> None:
    config_path = ROOT / "configs" / "fate_oia_train_360x640_acpr_meter_oia_v1.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert data["runtime"]["profile_candidates"] == [[16, 2], [12, 3], [8, 4], [6, 5]]
    assert data["runtime"]["hard_max_reserved_gb"] == 45.0
    assert data["splits"]["main_audit_calib_disjoint"] is True
    assert data["splits"]["audit_fraction"] > 0
    assert data["splits"]["calib_fraction"] > 0
    assert data["posthoc_calibration"]["fit_split"] == "train_calib"
    assert data["posthoc_calibration"]["updates_representation"] is False
