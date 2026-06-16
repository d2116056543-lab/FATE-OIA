from pathlib import Path

import yaml


def test_candidate_probe_config_blocks_stage_b_by_default():
    cfg_path = Path("configs/fate_oia_train_360x640_acpr_actalign_v1_3_candidate_probe.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["actalign"]["stage_mode"] == "candidate_probe"
    assert cfg["stageA"]["train_candidate_heads_only"] is True
    assert cfg["stageB"]["enabled"] is False
    assert cfg["stageB"]["require_stageA_pass"] is True
    assert cfg["runtime"]["foreground_only"] is True
    assert cfg["experiment"]["feature_cache_enabled"] is False
    assert cfg["experiment"]["token_compression"] == "none"

