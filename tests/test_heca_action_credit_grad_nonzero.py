import torch

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_action_objective_reaches_credit_parameters() -> None:
    module = StateConditionedActionCredit(dim=8, factor_dim=3, rank=4)
    out = module(
        torch.zeros(2, 4), torch.randn(2, 4, 8), torch.randn(2, 3, 8),
        torch.softmax(torch.randn(2, 3, 3), -1), torch.ones(2, 3), torch.ones(3), progress=1.0,
    )
    out["action_logits_final"].sum().backward()
    assert module.learned_action_factor_bias.grad is not None
    assert module.state_effect_embedding.grad is not None
    assert module.state_effect_embedding.grad.abs().sum() > 0

