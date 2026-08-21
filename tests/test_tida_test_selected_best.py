import yaml


def test_primary_best_is_internal_test_deploy_joint():
    cfg = yaml.safe_load(open("configs/fate_oia_train_tida_oia_v1_15f.yaml", encoding="utf-8"))
    assert cfg["experiment"]["best_selection_split"] == "test"
    assert cfg["experiment"]["best_selection_metric"] == "deploy_joint"
    assert cfg["experiment"]["internal_test_selected"] is True
    assert cfg["experiment"]["publication_eligible"] is False
