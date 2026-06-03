from __future__ import annotations

from pathlib import Path

import yaml

from fate_oia.engine.audit_psr_train_oia_implementation import check_config, check_source_static


def test_psr_train_config_is_test_only_and_no_cache():
    failures = check_config(Path("configs/fate_oia_train_360x640_psr_train_oia_v1.yaml"))
    assert failures == []


def test_psr_train_sources_do_not_use_old_logits_or_background_supervisor():
    failures = check_source_static()
    assert failures == []


def test_psr_train_config_has_expected_protocol_keys():
    cfg = yaml.safe_load(Path("configs/fate_oia_train_360x640_psr_train_oia_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["test_only_evaluation"] is True
    assert cfg["best_selection_split"] == "test"
    assert cfg["feature_cache_enabled"] is False
    assert cfg["old_logits_training"] is False
    assert cfg["batch_size"] * cfg["gradient_accumulation_steps"] == 32
