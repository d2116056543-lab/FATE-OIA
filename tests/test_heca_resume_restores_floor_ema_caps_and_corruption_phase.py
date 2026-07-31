import torch

from fate_oia.optim.heca_optimization import HECAExcessRiskBalancer, HECAScheduleState, ReasonProbabilityEMA


def test_resume_restores_all_adaptive_state() -> None:
    schedule = HECAScheduleState(update=13, total_updates=80, corruption_phase=2, foundation_grad_ema=1.7, action_floor=0.2, reason_floor=0.3)
    restored = HECAScheduleState.from_state_dict(schedule.state_dict())
    assert restored.state_dict() == schedule.state_dict()
    balancer = HECAExcessRiskBalancer()
    balancer.update_floors(torch.tensor(0.4), torch.tensor(0.6))
    other = HECAExcessRiskBalancer()
    other.load_state_dict(balancer.state_dict())
    assert other.state_dict() == balancer.state_dict()
    ema = ReasonProbabilityEMA()
    ema.update(["x"], torch.full((1, 21), 0.4))
    copied = ReasonProbabilityEMA()
    copied.load_state_dict(ema.state_dict())
    assert torch.equal(copied.values["x"], ema.values["x"])
