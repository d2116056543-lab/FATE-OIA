from pathlib import Path

from fate_oia.engine.audit_ceai_oia_implementation import load_config_flat, validate_config


def test_ceai_config_enforces_no_cache_test_only():
    cfg = load_config_flat(Path("configs/fate_oia_train_360x640_ceai_oia_v1.yaml"))
    errors = validate_config(cfg)
    assert errors == []
    assert cfg["feature_cache_enabled"] is False
    assert cfg["token_compression"] == "none"
    assert cfg["test_only_evaluation"] is True
    assert cfg["best_selection_split"] == "test"
