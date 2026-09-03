from pathlib import Path

import yaml


CONFIG = Path("configs/fate_oia_train_tida_target_difference_v29.yaml")


def test_v29_enables_only_direct_target_conditioned_temporal_route():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = config["model"]
    loss = config["loss"]

    assert config["experiment"]["name"] == "tida_target_difference_v29"
    assert model["target_token_flow_enabled"] is True
    assert model["target_token_direct_difference_enabled"] is True
    assert model["target_token_flow_independent_scale"] is True
    assert model["legacy_semantic_routes_enabled"] is False
    assert model["logit_flow_enabled"] is False
    assert model["geometric_flow_enabled"] is False
    assert model["traffic_action_enabled"] is False
    assert model["traffic_trajectory_enabled"] is False
    assert model["relational_traffic_enabled"] is False
    assert model["reason_local_query_enabled"] is False
    assert model["action_local_query_enabled"] is False
    assert model["object_intent_enabled"] is False
    assert loss["target_token_action_prediction"] == 0.0
    assert loss["target_token_reason_prediction"] == 0.0
    assert loss["target_token_action_aux"] > 0
    assert loss["target_token_reason_aux"] > 0
    assert loss["target_token_action_order"] > 0
    assert loss["target_token_reason_order"] > 0
