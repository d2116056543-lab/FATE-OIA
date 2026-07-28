import torch
from torch import nn

from fate_oia.optim.meter_meta_utility import METERMetaUtility, shadow_adamw_update


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
        reason_loss_fn=lambda values, factor_id: (
            values["up"][factor_id] - 2.0
        ).square(),
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
        reason_loss_fn=lambda values, factor_id: values["up"][factor_id].square(),
        audit_action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
    )
    assert event.action_reason_loss > event.action_only_loss
    assert float(event.omega_after[0]) <= float(event.omega_before[0])


def test_meta_utility_evaluates_heldout_candidates_without_building_graphs() -> None:
    utility = METERMetaUtility(
        factors=1,
        virtual_lr=0.1,
        ema_old_weight=0.0,
        ema_new_weight=1.0,
    )
    grad_enabled: list[bool] = []

    def heldout(values: dict[str, torch.Tensor]) -> torch.Tensor:
        grad_enabled.append(torch.is_grad_enabled())
        return (values["up"] - 2.0).square()

    utility.event(
        {"up": torch.tensor([1.0])},
        factor_ids=(0,),
        action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
        reason_loss_fn=lambda values, factor_id: (
            values["up"][factor_id] - 2.0
        ).square(),
        audit_action_loss_fn=heldout,
    )

    assert grad_enabled
    assert not any(grad_enabled)


def test_meta_utility_uses_factor_specific_reason_loss() -> None:
    utility = METERMetaUtility(
        factors=2,
        virtual_lr=0.1,
        ema_old_weight=0.0,
        ema_new_weight=1.0,
        lower=-0.01,
        upper=0.01,
    )
    parameter = {"up": torch.tensor([1.0, 1.0])}
    requested_factor_ids: list[int] = []

    def reason_loss(values: dict[str, torch.Tensor], factor_id: int) -> torch.Tensor:
        requested_factor_ids.append(factor_id)
        target = torch.tensor([2.0, 0.0])
        return (values["up"][factor_id] - target[factor_id]).square()

    event = utility.event(
        parameter,
        factor_ids=(0, 1),
        action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
        reason_loss_fn=reason_loss,
        audit_action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
    )

    assert requested_factor_ids == [0, 1]
    assert float(event.relative_utility[0]) > 0.0
    assert float(event.relative_utility[1]) < 0.0


def test_meta_utility_reports_unresolvable_float32_candidate() -> None:
    utility = METERMetaUtility(
        factors=1,
        virtual_lr=1e-4,
        ema_old_weight=0.0,
        ema_new_weight=1.0,
        lower=0.001,
        upper=0.005,
    )
    parameter = {"up": torch.tensor([1.0], dtype=torch.float32)}

    event = utility.event(
        parameter,
        factor_ids=(0,),
        action_loss_fn=lambda values: 1e-6 * (values["up"] - 2.0).square().sum(),
        reason_loss_fn=lambda values, factor_id: 1e-6
        * (values["up"][factor_id] - 2.0).square(),
        audit_action_loss_fn=lambda values: (
            0.8 + 1e-6 * (values["up"] - 2.0).square().sum()
        ).to(torch.float32),
    )

    assert event.resolution_failure.tolist() == [True]
    assert torch.equal(event.relative_utility, torch.zeros_like(event.relative_utility))
    assert torch.isfinite(event.relative_utility).all()
    assert torch.equal(event.omega_before, event.omega_after)


def test_shadow_adamw_update_matches_real_optimizer_step() -> None:
    parameter = nn.Parameter(torch.tensor([[0.4, -0.2]], dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
    )
    parameter.grad = torch.tensor([[0.3, -0.7]])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    state = optimizer.state[parameter]
    gradient = torch.tensor([[-0.5, 0.25]])

    expected_parameter = nn.Parameter(before.clone())
    expected_optimizer = torch.optim.AdamW(
        [expected_parameter],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
    )
    expected_optimizer.state[expected_parameter] = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }
    expected_parameter.grad = gradient.clone()
    expected_optimizer.step()

    actual = shadow_adamw_update(
        before,
        gradient,
        exp_avg=state["exp_avg"],
        exp_avg_sq=state["exp_avg_sq"],
        step=state["step"],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
    )

    torch.testing.assert_close(actual, expected_parameter.detach())


def test_meta_utility_ema_is_bias_corrected_from_first_observation() -> None:
    utility = METERMetaUtility(
        factors=1,
        virtual_lr=0.1,
        ema_old_weight=0.9,
        ema_new_weight=0.1,
        lower=0.001,
        upper=0.005,
    )
    parameter = {"up": torch.tensor([1.0])}

    event = utility.event(
        parameter,
        factor_ids=(0,),
        action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
        reason_loss_fn=lambda values, factor_id: (
            values["up"][factor_id] - 2.0
        ).square(),
        audit_action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
    )

    assert event.observation_count.tolist() == [1]
    torch.testing.assert_close(
        event.utility_ema_bias_corrected[0],
        event.relative_utility[0],
    )


def test_meta_event_uses_shadow_adamw_without_mutating_state() -> None:
    utility = METERMetaUtility(
        factors=1,
        virtual_lr=1e-4,
        ema_old_weight=0.0,
        ema_new_weight=1.0,
        lower=-0.01,
        upper=0.01,
    )
    parameter = {"up": torch.tensor([[1.0]], dtype=torch.float32)}
    shadow_state = {
        "up": {
            "exp_avg": torch.tensor([[0.2]]),
            "exp_avg_sq": torch.tensor([[0.04]]),
            "step": torch.tensor(5.0),
            "lr": 1e-2,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
    }
    parameter_before = parameter["up"].clone()
    state_before = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in shadow_state["up"].items()
    }

    event = utility.event(
        parameter,
        factor_ids=(0,),
        action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
        reason_loss_fn=lambda values, factor_id: (
            values["up"][factor_id] - 2.0
        ).square().sum(),
        audit_action_loss_fn=lambda values: (values["up"] - 2.0).square().sum(),
        shadow_optimizer_state=shadow_state,
    )

    assert event.shadow_update_used
    assert event.candidate_delta_norm > 0.0
    assert torch.equal(parameter["up"], parameter_before)
    for key, before in state_before.items():
        after = shadow_state["up"][key]
        if isinstance(before, torch.Tensor):
            assert torch.equal(after, before)
        else:
            assert after == before
