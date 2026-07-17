from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule


def test_hidden_pu_failure_only_disables_pu():
    schedule = ICDORAdaptiveSchedule(pilot=True)
    schedule.set_pu_enabled(False, reason="hidden_margin_negative")
    phase = schedule.phase()
    assert phase.route_mode == "shadow"
    assert phase.latent_enabled is True
    assert phase.pu_enabled is False

