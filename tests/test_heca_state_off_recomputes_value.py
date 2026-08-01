import torch

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_uniform_state_recomputes_values_and_contributions() -> None:
    torch.manual_seed(11)
    module = StateConditionedActionCredit(dim=8, factor_dim=2, rank=4)
    with torch.no_grad():
        module.state_effect_embedding[:, 0, 0] = 1.0
        module.state_effect_embedding[:, 1, 0] = -1.0
    common = dict(
        action_logits_visual=torch.zeros(1, 4), action_nodes=torch.randn(1, 4, 8),
        factor_action_bridge_token=torch.randn(1, 2, 8), factor_reliability=torch.ones(1, 2),
        factor_action_ownership=torch.ones(2), progress=1.0,
    )
    state = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    clean = module(**common, factor_state_prob_credit=state)
    uniform = module(**common, factor_state_prob_credit=torch.full_like(state, 1 / 3))
    assert not torch.allclose(clean["action_factor_values"], uniform["action_factor_values"])
    assert not torch.allclose(clean["action_factor_contribution"], uniform["action_factor_contribution"])
