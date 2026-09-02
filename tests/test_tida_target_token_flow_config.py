from pathlib import Path

import yaml


CONFIG = Path("configs/fate_oia_train_tida_target_token_flow_v27.yaml")


def test_v27_config_is_target_private_15_frame_strong_baseline_training():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["data"]["history_frames"] == 14
    assert config["data"]["test_count"] == 4572
    assert config["data"]["num_workers"] == 4
    assert config["data"]["persistent_workers"] is True
    assert config["data"]["prefetch_factor"] >= 3
    assert config["data"]["history_unavailable_fallback"] == "exact_terminal_image"
    assert config["data"]["history_unavailable_expected_count"] == 35
    assert config["image_base"]["checkpoint"].endswith(
        "vetra_replay_from_scratch_v2_full_20260819_retry1\\checkpoint_stage_b_continued.pth"
    )
    assert config["image_base"]["stage_c_deployment"].endswith(
        "deploy_final_train_only\\vetra_from_scratch_deploy.pth"
    )
    assert config["image_base"]["checkpoint_sha256"] == (
        "720cae7af5836921f16599be5d980cf5576772b29ad475b06373b11733122415"
    )
    assert config["image_base"]["stage_c_deployment_sha256"] == (
        "11bf799c83750612cea1b089b548371a5c5af673af474b8a73124e80dac9d821"
    )
    assert config["image_base"]["expected_stage_c_metrics"] == {
        "Act_mF1": 0.7311010361,
        "Act_oF1": 0.7538076043,
        "Act_mAP": 0.7991195023,
        "Exp_mF1": 0.4058934152,
        "Exp_oF1": 0.5731635690,
        "Exp_mAP": 0.3846499542,
        "joint": 0.5684972256,
    }

    model = config["model"]
    assert model["target_token_flow_enabled"] is True
    for key in (
        "legacy_semantic_routes_enabled", "logit_flow_enabled",
        "geometric_flow_enabled", "traffic_action_enabled",
        "traffic_trajectory_enabled", "relational_traffic_enabled",
        "reason_local_query_enabled", "action_local_query_enabled",
        "traffic_adaptive_boundary_enabled", "object_intent_enabled",
    ):
        assert model[key] is False

    training = config["training"]
    assert training["epochs"] == 10
    assert training["batch_size"] == 6
    assert training["gradient_accumulation_steps"] == 5
    assert training["target_token_predictive_epochs"] == 2
    assert training["formal_train_owners"] == [
        "target_token_action", "target_token_reason"
    ]
    assert config["runtime"]["test_every_epoch"] is True
    assert config["runtime"]["no_feature_cache"] is True
    assert config["backbone"]["token_compression"] == "none"


def test_v27_config_has_complete_target_loss_and_train_only_policy():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    loss = config["loss"]
    for branch in ("action", "reason"):
        for suffix in ("prediction", "order", "aux", "rank", "utility", "no_harm", "delta"):
            assert f"target_token_{branch}_{suffix}" in loss
    deployment = config["deployment"]
    assert deployment["calibration_source"] == "train_calib"
    assert deployment["test_oracle_writeback"] is False
    assert 0.0 in deployment["target_token_action_policy_scales"]
    assert 0.0 in deployment["target_token_reason_policy_scales"]
