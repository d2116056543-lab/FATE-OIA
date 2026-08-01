import torch

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_nonlatent_left_factor_has_a_trainable_route_to_left_action() -> None:
    module = StateConditionedActionCredit(dim=8, factor_dim=21, rank=4)
    output = module(
        action_logits_visual=torch.zeros(2, 4),
        action_nodes=torch.randn(2, 4, 8),
        factor_action_bridge_token=torch.randn(2, 21, 8),
        factor_state_prob_credit=torch.softmax(torch.randn(2, 21, 3), -1),
        factor_reliability=torch.ones(2, 21),
        factor_action_ownership=torch.tensor([1.0] * 14 + [0.0] + [1.0] * 5 + [0.0]),
        progress=1.0,
    )
    loss = output["action_factor_contribution"][:, 2, 9].sum()
    loss.backward()
    assert module.state_effect_embedding.grad is not None
    assert module.state_effect_embedding.grad[9].abs().sum() > 0
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = module(
        action_logits_visual=torch.zeros(2, 4),
        action_nodes=torch.randn(2, 4, 8),
        factor_action_bridge_token=torch.randn(2, 21, 8),
        factor_state_prob_credit=torch.softmax(torch.randn(2, 21, 3), -1),
        factor_reliability=torch.ones(2, 21),
        factor_action_ownership=torch.tensor([1.0] * 14 + [0.0] + [1.0] * 5 + [0.0]),
        progress=1.0,
    )
    output["action_factor_contribution"][:, 2, 9].sum().backward()
    assert module.learned_action_factor_bias.grad[2, 9].abs() > 0
