from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_config_preserves_complete_core_path_and_fast_loader_contract():
    config = yaml.safe_load((ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8"))
    assert config["token_compression"] == "none"
    assert config["feature_cache_enabled"] is False
    assert config["training"]["num_workers"] == 8
    assert config["training"]["persistent_workers"] is True
    assert config["training"]["prefetch_factor"] == 4
    assert config["runtime"]["hard_max_reserved_gb"] == 46.5
