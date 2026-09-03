from pathlib import Path

import yaml


CONFIG = Path("configs/fate_oia_train_tida_target_token_flow_v28.yaml")


def test_v28_keeps_the_strong_image_contract_and_uses_non_self_cancelling_flow():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = config["model"]

    assert config["experiment"]["name"] == "tida_target_token_flow_v28"
    assert config["image_base"]["expected_stage_c_metrics"]["Act_mF1"] == 0.7311010361
    assert model["target_token_flow_enabled"] is True
    assert model["target_token_flow_innovation_weight"] == 0.25
    assert model["target_token_flow_motion_weight"] == 0.50
    assert model["target_token_flow_order_weight"] == 0.50
    assert 19.5 < model["target_token_flow_candidate_temperature"] < 19.7
    assert config["deployment"]["target_token_action_policy_allow_proper_score_tie"] is True
    assert config["deployment"]["target_token_reason_policy_allow_proper_score_tie"] is False
    assert config["training"]["target_token_predictive_epochs"] == 0
    assert config["runtime"]["no_feature_cache"] is True
    assert config["backbone"]["token_compression"] == "none"
