import torch

from fate_oia.losses.tida_losses import action_route_sparse_loss, terminal_gain_loss


def test_violation_losses_are_finite_and_positive():
    gain = terminal_gain_loss(torch.ones(2, 3), torch.zeros(2, 3), margin=0.03)
    route = torch.zeros(2, 4, 37); route[..., -1] = 1.0
    keys = torch.randn(2, 37, 8)
    sparse = action_route_sparse_loss(route, keys, valid_rho=torch.ones(2, dtype=torch.bool))
    assert torch.isfinite(gain) and gain.item() > 0
    assert torch.isfinite(sparse) and sparse.item() > 0


def test_route_sparse_near_null_diversity_gradient_is_bounded():
    # A nearly all-null route has no trustworthy action centroid yet. Comparing
    # its direction must not amplify a tiny non-null mass into a huge gradient.
    route = torch.zeros(2, 4, 37)
    route[..., 0] = 1e-10
    route[:, :, 1] = torch.tensor([0.0, 0.01, 0.02, 0.03]) * 1e-10
    route[..., -1] = 1.0 - route[..., :-1].sum(-1)
    route.requires_grad_()
    keys = torch.zeros(2, 37, 8)
    keys[..., 0, 0] = 1.0
    keys[..., 1, 1] = 1.0

    loss = action_route_sparse_loss(route, keys, valid_rho=torch.ones(2, dtype=torch.bool))
    loss.backward()

    assert torch.isfinite(loss)
    assert route.grad is not None and torch.isfinite(route.grad).all()
    assert route.grad.abs().max().item() < 100.0
