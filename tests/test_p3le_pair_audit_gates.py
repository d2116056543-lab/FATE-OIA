from pathlib import Path

from fate_oia.engine.audit_p3le_pair_oia_implementation import check_config, check_model, check_static


def test_audit_static_and_model_gates_have_no_failures():
    assert check_config(Path("configs/fate_oia_train_360x640_p3le_pair_oia_v1.yaml")) == []
    assert check_static() == []
    assert check_model() == []
