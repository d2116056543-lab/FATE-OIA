import torch
from torch import nn

from fate_oia.optim.meter_meta_utility import METERMetaUtility


def test_meta_utility_virtual_event_does_not_mutate_real_parameter() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    utility = METERMetaUtility(factors=2)
    before = parameter.detach().clone()
    event = utility.event(parameter, lambda value: (value - 2.0).square().sum(), (0,))
    assert torch.equal(parameter.detach(), before)
    assert event.factor_ids == (0,)


def test_meta_utility_increases_omega_for_beneficial_reason_gradient() -> None:
    utility = METERMetaUtility(factors=1, virtual_lr=0.1, ema_old_weight=0.0, ema_new_weight=1.0, lower=-0.01, upper=0.01)
    parameter = {"up": torch.tensor([1.0])}
    event = utility.event(
        parameter,
        factor_ids=(0,),
        action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
        reason_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
        audit_action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
    )
    assert event.action_reason_loss < event.action_only_loss
    assert float(event.omega_after[0]) > float(event.omega_before[0])


def test_meta_utility_rejects_harmful_reason_gradient() -> None:
    utility = METERMetaUtility(factors=1, virtual_lr=0.1, ema_old_weight=0.0, ema_new_weight=1.0, lower=-0.01, upper=0.01)
    parameter = {"up": torch.tensor([1.0])}
    event = utility.event(
        parameter,
        factor_ids=(0,),
        action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
        reason_loss_fn=lambda values: (values["up"] - 0.0).square().sum(),
        audit_action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
    )
    assert event.action_reason_loss > event.action_only_loss
    assert float(event.omega_after[0]) <= float(event.omega_before[0])
