from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule


def test_shadow_route_is_learning_active_before_certificate():
    phase = ICDORAdaptiveSchedule(pilot=True).phase()
    assert phase.route_mode == "shadow"
    assert phase.latent_enabled is True
    assert phase.enable_factor_losses is True

