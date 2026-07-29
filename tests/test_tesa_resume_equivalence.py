from fate_oia.engine.tesa_diagnostics import mechanism_ramps


def test_ramps_resume_from_optimizer_step() -> None:
    assert mechanism_ramps(5, 100) == mechanism_ramps(5, 100)
    assert mechanism_ramps(0, 100) == (0.25, 0.0)
    assert mechanism_ramps(10, 100) == (1.0, 1.0)
