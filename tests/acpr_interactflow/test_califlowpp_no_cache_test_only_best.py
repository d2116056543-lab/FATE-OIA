from __future__ import annotations

import yaml


def test_formal_runtime_is_no_cache_test_only_and_test_best() -> None:
    with open("configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["data"]["feature_cache_enabled"] is False
    assert cfg["data"]["token_cache_enabled"] is False
    assert cfg["data"]["logit_cache_enabled"] is False
    assert cfg["evaluation"]["eval_splits"] == ["test"]
    assert cfg["evaluation"]["best_selector"]["primary"] == "joint"
    assert "val" not in cfg["evaluation"]["eval_splits"]
