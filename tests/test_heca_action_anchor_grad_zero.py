import torch

from fate_oia.models.meter_signed_factors import selective_credit_bridge


def test_action_credit_cannot_update_anchor_but_bridges_state_and_global() -> None:
    anchor = torch.randn(2, 3, 4, requires_grad=True)
    state = torch.randn(2, 3, 4, requires_grad=True)
    global_token = torch.randn(2, 3, 4, requires_grad=True)
    bridge = selective_credit_bridge(anchor, state, global_token, scale=0.05)
    bridge.sum().backward()
    assert anchor.grad is None or torch.count_nonzero(anchor.grad) == 0
    torch.testing.assert_close(state.grad, torch.full_like(state, 0.05))
    torch.testing.assert_close(global_token.grad, torch.full_like(global_token, 0.05))

