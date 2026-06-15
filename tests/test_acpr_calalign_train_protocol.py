from pathlib import Path

import yaml


def test_calalign_config_uses_deploy_fixed_primary_and_no_cache():
    cfg = yaml.safe_load(Path("configs/fate_oia_train_360x640_acpr_calalign_v1_2.yaml").read_text())

    assert cfg["threshold"]["enabled"] is True
    assert cfg["eval"]["primary_raw_branch"] == "deploy_fixed"
    assert cfg["eval"]["also_eval_base_fixed"] is True
    assert cfg["threshold"]["train_calib_fraction"] == 0.10
    assert cfg["threshold"]["base_detach"] is True
    assert cfg["feature_cache_enabled"] is False
    assert cfg["token_compression"] == "none"
    assert cfg["best_selection_split"] == "test"


def test_train_code_contains_train_calib_teacher_and_not_test_teacher():
    text = Path("fate_oia/engine/train_acpr_oia.py").read_text(encoding="utf-8")

    assert "make_train_calib_indices" in text
    assert "train_calib_loader" in text
    assert "collect_threshold_teacher" in text
    assert "threshold_head.update_teacher" in text
    assert "test_oracle" in text
    assert "copy_test_threshold" not in text
    assert "action_set_probs @ subset_membership" not in text
