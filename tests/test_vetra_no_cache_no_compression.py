import yaml


def test_probe_config_forbids_cache_and_compression():
    cfg=yaml.safe_load(open("configs/fate_oia_train_360x640_vetra_oia_v1_probe.yaml",encoding="utf-8"))
    assert cfg["experiment"]["feature_cache_enabled"] is False
    assert cfg["experiment"]["token_compression"] == "none"
    assert cfg["experiment"]["best_selection_split"] == "test"
