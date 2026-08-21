import torch

from fate_oia.losses.tida_losses import action_route_sparse_loss, terminal_gain_loss


def test_violation_losses_are_finite_and_positive():
    gain = terminal_gain_loss(torch.ones(2, 3), torch.zeros(2, 3), margin=0.03)
    route = torch.zeros(2, 4, 37); route[..., -1] = 1.0
    keys = torch.randn(2, 37, 8)
    sparse = action_route_sparse_loss(route, keys, valid_rho=torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(gain) and gain.item() > 0
    assert torch.isfinite(sparse) and sparse.item() > 0
