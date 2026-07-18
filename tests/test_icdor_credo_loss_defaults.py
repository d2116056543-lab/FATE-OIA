from __future__ import annotations

import yaml
from pathlib import Path


def test_credo_loss_defaults_match_the_declared_pilot_objective() -> None:
    with open("configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert config["loss"]["policy"] == {
        "action_visual": 1.00,
        "action_rank": 0.10,
        "action_shadow": 0.50,
        "reason_visual_observed": 1.00,
        "reason_observed_rank": 0.05,
        "reason_observation": 0.20,
        "reason_posterior_rank": 0.05,
        "factor_presence": 1.00,
    }
    assert config["loss"]["factor"] == {
        **config["loss"]["factor"],
        "presence_weight": 1.00,
        "visibility_weight": 0.50,
        "geometry_mask_weight": 0.10,
        "selective_contrastive_weight": 0.05,
        "view_consistency_weight": 0.015,
        "flip_equivariance_weight": 0.015,
        "prototype_occupancy_weight": 0.005,
        "prototype_repulsion_weight": 0.005,
        "allow_reason_anchors": False,
        "positive_anchor_weight": 0.00,
    }
    assert config["loss"]["reason"] == {
        "visual_observed_asl_weight": 1.00,
        "observed_asl_weight": 0.00,
        "observation_nll_weight": 0.20,
        "posterior_bce_weight": 0.00,
        "posterior_rank_weight": 0.00,
        "factor_latent_consistency_weight": 0.02,
        "escape_token_weight": 0.005,
        "propensity_regularization_weight": 0.00,
    }


def test_factor_total_has_no_plan_external_weak_negative_weight() -> None:
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert '+ 0.05 * factor["loss_factor_weak_negative"]' not in source
