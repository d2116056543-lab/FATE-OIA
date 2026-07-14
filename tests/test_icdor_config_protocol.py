from __future__ import annotations

from pathlib import Path

import yaml


def test_icdor_config_uses_three_isolated_lanes_and_no_legacy_state_action_path() -> None:
    config = yaml.safe_load(
        Path("configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml").read_text(encoding="utf-8")
    )

    assert config["experiment"]["direct_image"] is True
    assert config["experiment"]["eval_splits"] == ["test"]
    assert config["experiment"]["best_selection_metric"] == "deploy_fixed_joint"
    assert config["backbone"]["freeze_backbone"] is True
    assert config["backbone"]["feature_cache"] is False
    assert config["backbone"]["token_compression"] == "none"
    assert config["model"]["formal_class"] == "MOSAICTrustICDORModel"
    assert config["model"]["separate_visual_pyramids"] is True
    assert config["model"]["action_set_final"] is False
    assert "state_composer" not in config["model"]
    assert config["model"]["adapter"]["rank"] == 48
    assert config["model"]["adapter"]["rezero_init"] == 0.0
    assert config["model"]["typed_attention"] == {
        "anchors_per_factor": 2,
        "heads": 4,
        "point_samples": 4,
        "curve_samples": 16,
        "region_samples": 12,
    }
    assert config["training"]["epochs"] == 12
    assert config["data"]["split_seed"] == 20260713
    assert config["calibration"]["train_calib_only"] is True
    assert config["calibration"]["deploy_equation"] == "raw_minus_theta"
