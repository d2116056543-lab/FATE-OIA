from fate_oia.optim.heca_optimization import HECAScheduleState


def test_resume_restores_every_stateful_schedule_field() -> None:
    state = HECAScheduleState(update=17, total_updates=100, corruption_phase=2)
    state.foundation_grad_ema = 0.7
    state.action_floor = 0.2
    state.reason_floor = 0.3
    state.pu_pass_streak = [2] * 21
    restored = HECAScheduleState.from_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()
    assert restored.foundation_lr_multiplier(logit_rms=4.0) == state.foundation_lr_multiplier(logit_rms=4.0)

