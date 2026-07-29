from fate_oia.engine.audit_acpr_meter_oia import FORBIDDEN_FORMAL, REQUIRED_FILES


def test_tesa_audit_contract_lists_formal_sources_and_forbidden_v1_paths():
    assert "fate_oia/engine/train_acpr_meter_oia.py" in REQUIRED_FILES
    assert "fate_oia/models/meter_oia_model.py" in REQUIRED_FILES
    assert "fate_oia/engine/tesa_diagnostics.py" in REQUIRED_FILES
    assert "action_selector" in FORBIDDEN_FORMAL
    assert "reason_logits_local" in FORBIDDEN_FORMAL
    assert "factor_support_map" in FORBIDDEN_FORMAL
