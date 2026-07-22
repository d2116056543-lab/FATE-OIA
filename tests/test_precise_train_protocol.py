from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_train_protocol_is_test_only_with_no_metric_early_stop():
    config = yaml.safe_load((ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8"))
    assert config["eval_splits"] == "test"
    assert config["best_selection_split"] == "test"
    assert config["training"]["no_metric_early_stop"] is True
    assert config["pu"] == {"enabled": False, "weight": 0.0}
