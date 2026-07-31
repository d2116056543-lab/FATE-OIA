import torch

from fate_oia.models.meter_signed_factors import selective_credit_bridge


def test_selective_bridge_gradient_ratio_is_five_percent() -> None:
    anchor = torch.randn(1, 2, 3, requires_grad=True)
    state = torch.randn(1, 2, 3, requires_grad=True)
    global_token = torch.randn(1, 2, 3, requires_grad=True)
    credit_input = selective_credit_bridge(anchor, state, global_token, scale=0.05)
    grad_credit, grad_state = torch.autograd.grad(
        credit_input.square().sum(), (credit_input, state), retain_graph=True
    )
    ratio = grad_state.norm() / grad_credit[..., 3:6].norm().clamp_min(1e-8)
    torch.testing.assert_close(ratio, torch.tensor(0.05), atol=1e-6, rtol=1e-6)

