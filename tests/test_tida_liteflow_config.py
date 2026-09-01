from pathlib import Path

import yaml


def test_liteflow_config_excludes_every_history_dino_route():
    config = yaml.safe_load(
        Path("configs/fate_oia_train_tida_liteflow_v21.yaml").read_text(encoding="utf-8")
    )
    model = config["model"]
    assert model["history_encoder_mode"] == "terminal_repeat"
    assert model["geometric_flow_enabled"] is True
    assert model["object_intent_enabled"] is False
    assert model["object_intent_terminal_semantics_only"] is True
    assert model["object_tracker_mode"] == "geometric_flow"
    for key in (
        "traffic_action_enabled",
        "traffic_trajectory_enabled",
        "relational_traffic_enabled",
        "reason_local_query_enabled",
    ):
        assert model[key] is False
    assert config["data"]["history_frames"] == 5
    assert config["backbone"]["feature_cache_enabled"] is False
    assert config["backbone"]["token_compression"] == "none"
    assert config["deployment"]["calibration_source"] == "train_calib"
    assert config["deployment"]["test_oracle_writeback"] is False
    assert config["deployment"]["locked_image_threshold_source"] == "v7_1_train_calib"
    assert len(config["deployment"]["locked_image_thresholds"]) == 25
    for key in (
        "geometric_action_aux",
        "geometric_action_rank",
        "geometric_reason_aux",
        "geometric_reason_rank",
    ):
        assert config["loss"][key] > 0
    assert config["data"]["num_workers"] == 4
    assert all(
        value == 0.0
        for key, value in config["loss"].items()
        if key.startswith("object_intent_")
    )


def test_trainer_wires_liteflow_constructor_switches():
    source = Path("fate_oia/engine/train_tida_oia.py").read_text(encoding="utf-8")
    assert 'history_encoder_mode=str(config["model"].get("history_encoder_mode", "dino"))' in source
    assert "object_intent_terminal_semantics_only=bool(" in source
