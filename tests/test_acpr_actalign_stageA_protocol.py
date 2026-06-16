from pathlib import Path

import yaml


def test_candidate_probe_config_blocks_stage_b_by_default():
    cfg_path = Path("configs/fate_oia_train_360x640_acpr_actalign_v1_3_candidate_probe.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["actalign"]["stage_mode"] == "candidate_probe"
    assert cfg["stageA"]["train_candidate_heads_only"] is True
    assert cfg["stageB"]["enabled"] is False
    assert cfg["stageB"]["require_stageA_pass"] is True
    assert cfg["training"]["lr_action_candidate"] == 0.0005
    assert cfg["candidate_probe"]["candidate_weight"] == 0.5
    assert cfg["candidate_probe"]["nonreg_weight"] == 0.5
    assert cfg["candidate_probe"]["gate_ema"] == 0.20
    assert cfg["candidate_probe"]["allow_reason_candidate"] is True
    assert cfg["candidate_probe"]["allow_predicate_candidate"] is True
    assert cfg["stageA"]["action_primary_score_action_weight"] == 0.85
    assert cfg["stageA"]["action_primary_score_exp_weight"] == 0.15
    assert cfg["model"]["final_action_source"] == "fallback_until_gate_selected"
    assert cfg["eval"]["primary_raw_branch"] == "fallback_until_gate_selected"
    assert cfg["runtime"]["foreground_only"] is True
    assert cfg["experiment"]["feature_cache_enabled"] is False
    assert cfg["experiment"]["token_compression"] == "none"


def test_candidate_probe_trainer_contains_hard_pass_gates():
    text = Path("fate_oia/engine/train_acpr_oia.py").read_text(encoding="utf-8")
    for token in [
        "gated_train_calib_act_gain",
        "test_act_nonregression",
        "test_exp_nonregression",
        "action_candidate_train_calib.jsonl",
        "--load_candidate_gate",
        "Clean full train is blocked",
    ]:
        assert token in text
