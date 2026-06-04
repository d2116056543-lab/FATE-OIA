from pathlib import Path

from fate_oia.engine.audit_p3le_pair_oia_implementation import check_config, check_model, check_static


def test_v1_1_audit_static_gates_are_strict_and_clean():
    assert check_config(Path("configs/fate_oia_train_360x640_p3le_pair_oia_v1.yaml")) == []
    assert check_static() == []
    assert check_model() == []

