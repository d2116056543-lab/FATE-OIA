from __future__ import annotations

import torch

from fate_oia.models.mosaic_low_rank_rezero_adapter import MOSAICLowRankReZeroPyramidAdapter


def _pyramid(dim: int = 8) -> dict[str, torch.Tensor]:
    return {
        "F_hi": torch.randn(2, dim, 5, 7),
        "F_mid": torch.randn(2, dim, 3, 4),
        "F_ctx": torch.randn(2, dim, 2, 3),
    }


def test_zero_output_initialization_preserves_each_pyramid_tensor_exactly() -> None:
    adapter = MOSAICLowRankReZeroPyramidAdapter(dim=8, rank=3, dropout=0.0)
    pyramid = _pyramid()
    adapted = adapter(pyramid)

    for name, original in pyramid.items():
        assert torch.equal(adapted[name], original)


def test_zero_output_adapter_is_not_a_dead_gradient_path() -> None:
    torch.manual_seed(4)
    adapter = MOSAICLowRankReZeroPyramidAdapter(dim=8, rank=3, dropout=0.0)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.2)
    pyramid = _pyramid()

    first_loss = sum(value.square().mean() for value in adapter(pyramid).values())
    first_loss.backward()
    assert adapter.blocks["F_hi"].up.weight.grad is not None
    assert adapter.blocks["F_hi"].up.weight.grad.abs().sum() > 1e-8
    assert adapter.blocks["F_hi"].depthwise.weight.grad is not None
    assert adapter.blocks["F_hi"].depthwise.weight.grad.abs().sum() > 1e-8
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = sum(value.square().mean() for value in adapter(pyramid).values())
    second_loss.backward()
    assert adapter.blocks["F_hi"].down.weight.grad is not None
    assert adapter.blocks["F_hi"].down.weight.grad.abs().sum() > 1e-8


def test_lane_parameters_are_independent_and_action_backward_does_not_touch_other_lanes() -> None:
    factor = MOSAICLowRankReZeroPyramidAdapter(dim=8, rank=3, dropout=0.0)
    action = MOSAICLowRankReZeroPyramidAdapter(dim=8, rank=3, dropout=0.0)
    reason = MOSAICLowRankReZeroPyramidAdapter(dim=8, rank=3, dropout=0.0)
    assert {id(parameter) for parameter in factor.parameters()}.isdisjoint(id(parameter) for parameter in action.parameters())
    assert {id(parameter) for parameter in action.parameters()}.isdisjoint(id(parameter) for parameter in reason.parameters())

    action_loss = sum(value.square().mean() for value in action(_pyramid()).values())
    action_loss.backward()
    assert all(parameter.grad is None for parameter in factor.parameters())
    assert all(parameter.grad is None for parameter in reason.parameters())
