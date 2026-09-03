from pathlib import Path

import yaml


CONFIG = Path("configs/fate_oia_train_tida_control_centered_difference_v30.yaml")


def test_v30_prioritizes_control_discrimination_over_static_auxiliary_fit():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = config["model"]
    loss = config["loss"]

    assert config["experiment"]["name"] == "tida_control_centered_difference_v30"
    assert model["target_token_direct_difference_enabled"] is True
    assert model["legacy_semantic_routes_enabled"] is False
    assert loss["target_token_action_prediction"] == 0.0
    assert loss["target_token_reason_prediction"] == 0.0
    assert loss["target_token_action_order"] >= loss["target_token_action_aux"]
    assert loss["target_token_reason_order"] >= loss["target_token_reason_aux"]
