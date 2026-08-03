import torch

from fate_oia.models.save_predicate_measurement import (
    ACTION_PREDICATE_BRIDGE_SCALE,
    SAVEPredicateMeasurement,
)


def test_save_action_predicate_bridge_derivative_is_fixed_five_percent() -> None:
    torch.manual_seed(13)
    measurement = SAVEPredicateMeasurement(dim=8)
    factor_nodes = torch.randn(1, 21, 8)
    patches = torch.randn(1, 3, 10, 8)
    output = measurement(factor_nodes, patches, progress=1.0)

    assert ACTION_PREDICATE_BRIDGE_SCALE == 0.05
    for raw_key, action_key in (
        ("predicate_map_raw", "predicate_map"),
        ("predicate_token_raw", "predicate_token"),
        ("predicate_state_prob_raw", "predicate_state_prob"),
        ("predicate_reliability_raw", "predicate_reliability"),
    ):
        gradient = torch.autograd.grad(
            output[action_key],
            output[raw_key],
            grad_outputs=torch.ones_like(output[action_key]),
            retain_graph=True,
        )[0]
        torch.testing.assert_close(
            gradient,
            torch.full_like(gradient, 0.05),
            atol=1e-7,
            rtol=1e-7,
        )


def test_save_action_route_parameter_gradient_is_single_five_percent_bridge() -> None:
    torch.manual_seed(17)
    measurement = SAVEPredicateMeasurement(dim=8)
    output = measurement(
        torch.randn(2, 21, 8),
        torch.randn(2, 3, 12, 8),
        progress=1.0,
    )
    parameter = measurement.typed_head.anchor_query.weight
    probe = torch.linspace(
        0.1,
        1.0,
        output["predicate_map_raw"].numel(),
    ).reshape_as(output["predicate_map_raw"])
    raw_gradient = torch.autograd.grad(
        (output["predicate_map_raw"] * probe).sum(),
        parameter,
        retain_graph=True,
    )[0]
    action_gradient = torch.autograd.grad(
        (output["predicate_map"] * probe).sum(),
        parameter,
    )[0]

    assert measurement.typed_head.action_measurement_grad_scale == 0.0
    assert raw_gradient.norm() > 0
    torch.testing.assert_close(
        action_gradient,
        ACTION_PREDICATE_BRIDGE_SCALE * raw_gradient,
        atol=1e-7,
        rtol=1e-6,
    )
    ratio = action_gradient.norm() / raw_gradient.norm()
    torch.testing.assert_close(
        ratio,
        torch.tensor(0.05),
        atol=1e-7,
        rtol=1e-6,
    )
