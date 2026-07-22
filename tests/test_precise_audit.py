from fate_oia.engine.audit_precise_oia_implementation import REQUIRED


def test_audit_requires_full_precise_source_surface():
    assert len(REQUIRED) >= 30
    assert "fate_oia/models/precise_oia_model.py" in REQUIRED
