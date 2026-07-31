import torch

from fate_oia.models.meter_signed_factors import selective_credit_bridge
from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_action_credit_cannot_update_anchor_but_bridges_state_and_global() -> None:
    anchor = torch.randn(2, 3, 4, requires_grad=True)
    state = torch.randn(2, 3, 4, requires_grad=True)
    global_token = torch.randn(2, 3, 4, requires_grad=True)
    bridge = selective_credit_bridge(anchor, state, global_token, scale=0.05)
    bridge.sum().backward()
    assert anchor.grad is None or torch.count_nonzero(anchor.grad) == 0
    torch.testing.assert_close(state.grad, torch.full_like(state, 0.05))
    torch.testing.assert_close(global_token.grad, torch.full_like(global_token, 0.05))


def test_action_credit_cannot_backprop_through_measurement_reliability() -> None:
    module = StateConditionedActionCredit(dim=8, action_dim=4, factor_dim=3, rank=4)
    visual = torch.randn(2, 4)
    action_nodes = torch.randn(2, 4, 8)
    bridge = torch.randn(2, 3, 8)
    state = torch.softmax(torch.randn(2, 3, 3), dim=-1)
    reliability_raw = torch.randn(2, 3, requires_grad=True)
    reliability = torch.sigmoid(reliability_raw)
    owner = torch.ones(3)

    output = module(visual, action_nodes, bridge, state, reliability, owner, progress=1.0)
    output["action_logits_final"].sum().backward()

    assert reliability_raw.grad is None or torch.count_nonzero(reliability_raw.grad) == 0
