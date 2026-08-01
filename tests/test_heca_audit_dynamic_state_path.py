from fate_oia.engine.audit_meter_oia_v3_heca import _dynamic_checks


def test_dynamic_audit_exercises_the_nonzero_state_value_path() -> None:
    checks = _dynamic_checks()["checks"]

    assert checks["state_uniform_recomputes_values"] is True
