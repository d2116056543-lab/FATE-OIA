from pathlib import Path

import yaml

from fate_oia.engine.profile_precise_oia import choose_runtime_profile


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_config_preserves_complete_core_path_and_fast_loader_contract():
    config = yaml.safe_load((ROOT / "configs" / "fate_oia_train_360x640_precise_oia_v1.yaml").read_text(encoding="utf-8"))
    assert config["token_compression"] == "none"
    assert config["feature_cache_enabled"] is False
    assert config["training"]["num_workers"] == 8
    assert config["training"]["persistent_workers"] is True
    assert config["training"]["prefetch_factor"] == 4
    assert config["runtime"]["hard_max_reserved_gb"] == 46.5


def test_runtime_profile_selects_fastest_safe_profile_and_prefers_lower_memory_on_ties():
    selected = choose_runtime_profile([
        {"batch_size": 10, "samples_per_sec": 5.0, "peak_reserved_gb": 45.5, "valid": True},
        {"batch_size": 8, "samples_per_sec": 4.9, "peak_reserved_gb": 40.0, "valid": True},
        {"batch_size": 6, "samples_per_sec": 6.0, "peak_reserved_gb": 47.0, "valid": False},
    ], hard_limit_gb=46.5)
    assert selected["batch_size"] == 8


def test_runtime_profile_binds_real_forward_to_git_and_config():
    source = (ROOT / "fate_oia" / "engine" / "profile_precise_oia.py").read_text(encoding="utf-8")
    assert 'selected["git_head"]' in source
    assert 'selected["config_sha256"]' in source
    assert 'root.parent / "real_forward.json"' in source
    assert "_observed_firewall(" in source
    assert "owner_gradient_matrix_passed" in source
    assert "curve_distance_valid_count" in source
    assert 'config["runtime"]["target_peak_reserved_gb"]' in source
