import yaml
from pathlib import Path


def test_fusionlite_config_protocol():
    cfg = yaml.safe_load(Path("configs/fate_oia_train_360x640_acpr_fusionlite_v1_4.yaml").read_text(encoding="utf-8"))
    assert cfg["eval_splits"] == "test"
    assert cfg["best_selection_split"] == "test"
    assert cfg["token_compression"] == "none"
    assert cfg["feature_cache_enabled"] is False
    assert cfg["model"]["use_fusionlite"] is True
    assert cfg["optim"]["epochs"] == 16
    assert cfg["loss_weights"]["fusionlite_delta_l2"] == 0.001
    assert cfg["loss_weights"]["r2a_forbidden_prior"] == 0.005
