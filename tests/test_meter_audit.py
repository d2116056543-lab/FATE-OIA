from fate_oia.engine.audit_acpr_meter_oia import REQUIRED_FILES


def test_audit_contract_lists_the_formal_source_files() -> None:
    assert "fate_oia/engine/train_acpr_meter_oia.py" in REQUIRED_FILES
    assert "fate_oia/models/meter_oia_model.py" in REQUIRED_FILES
    assert "fate_oia/optim/meter_meta_utility.py" in REQUIRED_FILES
