from pathlib import Path
import yaml


def test_actalign_config_protocol_and_artifacts_declared():
    cfg = yaml.safe_load(Path("configs/fate_oia_train_360x640_acpr_actalign_v1_3.yaml").read_text(encoding="utf-8"))
    assert cfg["feature_cache_enabled"] is False
    assert cfg["token_compression"] == "none"
    assert cfg["eval_splits"] == "test"
    assert cfg["best_selection_split"] == "test"
    assert cfg["model"]["action_set_affects_final_action"] is False
    assert cfg["model"]["graph_delta_to_logits"] is False
    assert cfg["actalign"]["max_pred_delta"] <= 0.05
    assert cfg["actalign"]["max_r2a_delta"] <= 0.20
    train_src = Path("fate_oia/engine/train_acpr_oia.py").read_text(encoding="utf-8")
    for token in ["action_primary_score", "action_utility_metrics.jsonl", "action_utility_gates.jsonl", "gradient_guard_stats.jsonl", "cooldown_stats.jsonl", "ema_swa_metrics.jsonl", "checkpoint_best_test_action_primary.pth"]:
        assert token in train_src
