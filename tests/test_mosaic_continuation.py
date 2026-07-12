from fate_oia.engine.mosaic_schedule import mosaic_continuation_phase_controls


def test_continuation_reopens_representation_learning():
    early = mosaic_continuation_phase_controls(0)
    late = mosaic_continuation_phase_controls(4)

    assert early.phase == "G_post_best_recovery"
    assert early.calibration_only is False
    assert early.posterior_enabled is True
    assert early.representation_lr_scale == 0.10
    assert late.action_state_gate_cap == 0.25
    assert late.reason_state_contribution_cap == 0.20
    assert late.representation_lr_scale == 0.20


def test_continuation_schedule_is_bounded():
    for epoch in range(5):
        assert mosaic_continuation_phase_controls(epoch).epoch == epoch

