import yaml


def test_runtime_contract_uses_bf16_and_safe_memory_ceiling():
    cfg = yaml.safe_load(open("configs/fate_oia_train_360x640_aie_oia_v1.yaml", encoding="utf-8"))
    assert cfg["training"]["precision"] == "bf16"
    assert cfg["runtime"]["max_reserved_memory_gb"] == 45.0
    assert cfg["experiment"]["feature_cache_enabled"] is False
    assert cfg["experiment"]["token_compression"] == "none"

