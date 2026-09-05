from pathlib import Path

import yaml

from fate_oia.losses.tida_loss_registry import TIDALossRegistry


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fate_oia_train_tida_action_conditioned_reason_v43.yaml"


def test_v43_uses_action_conditioned_reason_without_v42_directional_path():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    model = config["model"]
    assert config["experiment"]["name"] == "tida_action_token_conditioned_reason_v46"

    assert model["reason_local_action_condition_enabled"] is True
    assert model["reason_local_action_condition_mode"] == "token"
    assert model["reason_local_action_condition_input_scale"] == 20.0
    assert model["reason_local_action_condition_cap"] == 0.01
    assert config["loss"]["reason_local_action_condition_aux"] == 0.10
    assert config["loss"]["reason_local_action_condition_rank"] == 0.05
    assert model.get("reason_local_directional_enabled", False) is False
    assert model["action_local_query_enabled"] is True
    assert model["reason_local_query_enabled"] is True
    assert model["track_conditioned_local_query_enabled"] is True
    assert config["training"]["formal_train_owners"] == [
        "action_local_query",
        "reason_local_query",
    ]
    assert config["data"]["test_count"] == 4572
    assert config["backbone"]["feature_cache_enabled"] is False
    assert config["backbone"]["token_compression"] == "none"
    TIDALossRegistry(config["loss"])
