from fate_oia.models.meter_semantic_action import heca_credit_ramp
from fate_oia.optim.heca_optimization import correction_fraction_for_run


def test_credit_schedule_and_pilot_fraction_are_unambiguous() -> None:
    assert heca_credit_ramp(0.05) == 0
    assert 0.49 < heca_credit_ramp(0.125) < 0.51
    assert heca_credit_ramp(0.20) == 1
    assert correction_fraction_for_run("pilot", gate_c_pass=False) == 0.20
    assert correction_fraction_for_run("full", gate_c_pass=True) == 0.25

