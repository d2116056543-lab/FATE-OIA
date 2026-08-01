from __future__ import annotations

import torch

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def _inputs(dim: int = 4) -> dict[str, torch.Tensor | float]:
    return {
        "action_logits_visual": torch.randn(2, 4),
        "action_nodes": torch.randn(2, 4, dim),
        "factor_action_bridge_token": torch.randn(2, 2, dim),
        "factor_state_prob_credit": torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        ),
        "factor_reliability": torch.ones(2, 2),
        "factor_action_ownership": torch.ones(2),
        "progress": 1.0,
    }


def test_state_value_branch_is_zero_effect_at_initialization_but_has_two_step_gradients() -> None:
    torch.manual_seed(11)
    module = StateConditionedActionCredit(dim=4, factor_dim=2, rank=3, max_states=3)
    assert tuple(module.factor_value_weight.shape) == (2, 3, 4)
    assert tuple(module.action_value_weight.shape) == (4, 3, 4)
    assert module.state_effect_embedding.eq(0).all()

    inputs = _inputs()
    initial = module(**inputs)
    assert torch.allclose(
        initial["action_logits_final"], initial["action_logits_visual"], atol=1e-7
    )

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    initial["action_logits_final"].sum().backward()
    assert module.state_effect_embedding.grad is not None
    assert module.state_effect_embedding.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = module(**inputs)
    second["action_logits_final"].sum().backward()
    assert module.factor_value_weight.grad is not None
    assert module.action_value_weight.grad is not None
    assert module.factor_value_weight.grad.abs().sum() > 0
    assert module.action_value_weight.grad.abs().sum() > 0


def test_state_values_use_factor_specific_and_sample_conditioned_action_queries() -> None:
    module = StateConditionedActionCredit(dim=4, factor_dim=1, rank=2, max_states=3)
    with torch.no_grad():
        module.factor_value_weight.zero_()
        module.factor_value_weight[0, 0, 0] = 1.0
        module.factor_value_bias.zero_()
        module.action_value_weight.zero_()
        module.action_value_weight[0, 0, 0] = 1.0
        module.action_value_weight[1, 0, 0] = -1.0
        module.action_value_bias.zero_()
        module.state_effect_embedding.zero_()
        module.state_effect_embedding[0, 0, 0] = 1.0

    action_nodes = torch.zeros(1, 4, 4)
    action_nodes[0, 0, 0] = 1.0
    action_nodes[0, 1, 0] = 1.0
    out = module(
        action_logits_visual=torch.zeros(1, 4),
        action_nodes=action_nodes,
        factor_action_bridge_token=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        factor_state_prob_credit=torch.tensor([[[1.0, 0.0, 0.0]]]),
        factor_reliability=torch.ones(1, 1),
        factor_action_ownership=torch.ones(1),
        progress=1.0,
    )
    assert out["action_factor_values"][0, 0, 0] > 0
    assert out["action_factor_values"][0, 1, 0] < 0
