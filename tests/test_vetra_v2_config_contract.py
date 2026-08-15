from pathlib import Path

import yaml


def test_joint_and_refine_configs_enforce_training_contract():
    root = Path(__file__).parents[1] / "configs"
    joint = yaml.safe_load((root / "fate_oia_train_360x640_vetra_trainable_073_040_v2_joint.yaml").read_text())
    refine = yaml.safe_load((root / "fate_oia_train_360x640_vetra_trainable_073_040_v2_reason_refine.yaml").read_text())
    for config in (joint, refine):
        assert config["calibration"]["exclude_from_training"] is True
        assert config["data"]["train_on_all_train"] is False
        assert config["experiment"]["best_selection_split"] == "test"
        assert config["experiment"]["feature_cache_enabled"] is False
        assert config["experiment"]["token_compression"] == "none"
        assert config["loss_weights"]["final_reason_dr"] > 0
        assert config["reason_dr"]["gamma_pair"] > config["reason_dr"]["gamma_negative"] > 0
        assert config["ema"]["enabled"] is True
    assert refine["training"]["trainable_owners"] == ["reason_private"]
