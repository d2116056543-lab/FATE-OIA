from __future__ import annotations

import yaml

from fate_oia.engine.audit_califlowpp_current_branch import _coalesce_int


def test_formal_config_consumes_runtime_image_cache_eval_and_loss_fields() -> None:
    with open("configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    visual = cfg["model"]["visual_encoder"]
    assert visual["dino_input_height"] == 320
    assert visual["dino_input_width"] == 576
    assert visual["anchor_frames"] == [0, 3, 6, 9, 12, 14]
    assert visual["dino_chunk_size"] == 6
    assert cfg["data"]["feature_cache_enabled"] is False
    assert cfg["data"]["token_cache_enabled"] is False
    assert cfg["data"]["logit_cache_enabled"] is False
    assert cfg["data"]["formal_input_uses_target_frame"] is False
    assert cfg["evaluation"]["eval_splits"] == ["test"]
    assert cfg["evaluation"]["best_selector"]["primary"] == "joint"
    assert "test" in cfg["evaluation"]["eval_splits"]
    assert cfg["loss"]["action_final_soft_kl"] == 1.0
    assert cfg["loss"]["exp29_calibrated_asl"] == 0.25


def test_review_pass_profile_null_grad_accum_falls_back_to_config() -> None:
    assert _coalesce_int(None, "5", default=1) == 5
    assert _coalesce_int(None, None, default=6) == 6
    assert _coalesce_int("bad", 8, default=1) == 8
