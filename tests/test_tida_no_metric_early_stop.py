import yaml


def test_metric_early_stop_is_forbidden():
    cfg = yaml.safe_load(open("configs/fate_oia_train_tida_oia_v1_15f.yaml", encoding="utf-8"))
    assert cfg["training"]["epochs"] == 10
    assert cfg["training"]["no_metric_early_stop"] is True
