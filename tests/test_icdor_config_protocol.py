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
    assert config["loss"]["policy"]["action_rank"] == 0.10
    assert config["loss"]["policy"]["reason_posterior_rank"] == 0.05
    assert config["loss"]["action_route"]["shadow_asl_weight"] == 0.50
    assert config["loss"]["action_route"]["intervention_weight"] == 0.10
    assert config["loss"]["factor"]["visibility_weight"] == 0.50
    assert config["loss"]["factor"]["selective_contrastive_weight"] == 0.05
    assert config["loss"]["factor"]["positive_anchor_weight"] == 0.00
    assert config["loss"]["reason"]["factor_latent_consistency_weight"] == 0.02
    assert config["loss"]["reason"]["escape_token_weight"] == 0.005
    assert config["edge_admission"]["audit_refresh_epochs"] == 2
    assert "build_epoch" not in config["factor_certificate"]
    assert "build_epoch" not in config["edge_admission"]
    assert "require_certificate_tier" not in config["edge_admission"]
