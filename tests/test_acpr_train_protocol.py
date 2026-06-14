import yaml


def test_acpr_config_protocol():
    cfg = yaml.safe_load(open("configs/fate_oia_train_360x640_acpr_oia_v1.yaml", encoding="utf-8"))
    assert cfg["eval_splits"] == "test"
    assert cfg["best_selection_split"] == "test"
    assert cfg["token_compression"] == "none"
    assert cfg["feature_cache_enabled"] is False
    assert cfg["training"]["epochs"] == 28
    assert cfg["training"]["fallback_ladder"] == [[6,5], [5,6], [4,8], [3,11]]
