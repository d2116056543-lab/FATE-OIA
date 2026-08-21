import yaml

from fate_oia.utils.tida_contracts import validate_training_protocol


def test_config_evaluates_only_test_every_epoch():
    cfg = yaml.safe_load(open("configs/fate_oia_train_tida_oia_v1_15f.yaml", encoding="utf-8"))
    validate_training_protocol(cfg)
    assert cfg["experiment"]["eval_splits"] == ["test"]
    assert cfg["runtime"]["test_every_epoch"] is True
