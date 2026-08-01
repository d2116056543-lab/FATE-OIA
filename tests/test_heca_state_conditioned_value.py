import torch

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_state_probability_changes_factor_value_not_only_route() -> None:
    torch.manual_seed(4)
    module = StateConditionedActionCredit(dim=8, factor_dim=2, rank=4, max_states=3)
    with torch.no_grad():
        module.state_effect_embedding[0, 0, 0] = 1.0
        module.state_effect_embedding[0, 1, 0] = -1.0
    common = dict(
        action_logits_visual=torch.zeros(1, 4),
        action_nodes=torch.randn(1, 4, 8),
        factor_action_bridge_token=torch.randn(1, 2, 8),
        factor_reliability=torch.ones(1, 2),
        factor_action_ownership=torch.ones(2),
        progress=1.0,
    )
    first = module(**common, factor_state_prob_credit=torch.tensor([[[1., 0., 0.], [1., 0., 0.]]]))
    second = module(**common, factor_state_prob_credit=torch.tensor([[[0., 1., 0.], [0., 1., 0.]]]))
    assert not torch.allclose(first["action_factor_state_values"], second["action_factor_state_values"])
    assert not torch.allclose(first["action_factor_values"], second["action_factor_values"])
