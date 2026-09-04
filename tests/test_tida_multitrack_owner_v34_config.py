from pathlib import Path

import yaml


def test_v34_adds_zero_init_multi_track_credit_without_rank_memory() -> None:
    config_path = Path(__file__).parents[1] / "configs" / (
        "fate_oia_train_tida_multitrack_owner_v34.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["experiment"]["name"] == "tida_multitrack_owner_v34"
    model = config["model"]
    loss = config["loss"]
    training = config["training"]

    assert model["traffic_trajectory_enabled"] is True
    assert model["traffic_trajectory_deploy_enabled"] is True
    assert model["traffic_trajectory_credit_mode"] == "ordered_vs_static"
    assert model["traffic_trajectory_static_utility_open_prior"] == 0.10
    assert model["traffic_trajectory_multi_track_enabled"] is True
    assert model["traffic_trajectory_multi_track_floor"] == 0.10
    assert model["traffic_trajectory_state_enabled"] is False
    assert model["traffic_trajectory_state_strength_scale"] == 8.0
    assert model["traffic_trajectory_state_cap_ratio"] == 0.5
    assert model["traffic_trajectory_state_utility_open_prior"] == 0.10
    assert model["action_patch_selection"] == "contrastive_diverse"
    assert model["action_patch_nms_radius"] == 2
    assert model["action_patch_specificity_power"] == 1.5
    assert model["target_token_flow_enabled"] is False
    assert model["object_intent_enabled"] is False
    assert model["traffic_adaptive_boundary_enabled"] is False
    assert model["relational_traffic_enabled"] is False
    assert training["formal_train_owners"] == [
        "traffic_trajectory", "traffic_trajectory_utility"
    ]
    assert training["action_rank_memory_capacity"] == 0
    assert loss["trajectory_action_boundary"] > 0
    assert loss["trajectory_action_rank"] > 0
    assert loss["trajectory_selected_control"] > 0
    assert loss["trajectory_utility_calibration"] > 0
    assert all(
        value == 0
        for name, value in loss.items()
        if name.startswith("target_token_")
    )
