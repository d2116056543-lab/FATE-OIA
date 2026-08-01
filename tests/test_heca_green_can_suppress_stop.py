import torch

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_same_green_factor_can_learn_opposite_forward_and_stop_values() -> None:
    module = StateConditionedActionCredit(dim=4, factor_dim=1, rank=2, max_states=3)
    with torch.no_grad():
        module.action_value_weight.zero_()
        module.action_value_weight[0, 0, 0] = 1
        module.action_value_weight[1, 0, 0] = -1
        module.action_value_bias.zero_()
        module.state_effect_embedding.zero_()
        module.state_effect_embedding[0, 0, 0] = 1
        module.factor_value_weight.zero_()
        module.factor_value_weight[0, 0, 0] = 1
        module.factor_value_bias.zero_()
    action_nodes = torch.zeros(1, 4, 4)
    action_nodes[0, 0, 0] = 1
    action_nodes[0, 1, 0] = 1
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
