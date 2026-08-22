import torch

from fate_oia.models.tida_action_reader import TIDAActionReader


def test_factor_contributions_reconstruct_bounded_delta():
    torch.manual_seed(1)
    model = TIDAActionReader(dim=16, num_actions=4, num_predicates=32, kappa=0.15)
    out = model(torch.randn(3, 4, 16), torch.randn(3, 32, 16), torch.randn(3, 4, 16), torch.rand(3, 36), temporal_scale=1.0)
    reconstructed = out["action_factor_contribution"].sum(-1)
    assert torch.allclose(reconstructed, out["action_temporal_delta"], atol=1e-6)


def test_zero_initialized_action_output_has_finite_first_open_gradient():
    torch.manual_seed(20260821)
    model = TIDAActionReader(dim=16, num_actions=4, num_predicates=32, kappa=0.15)
    out = model(
        torch.randn(3, 4, 16),
        torch.randn(3, 32, 16),
        torch.randn(3, 4, 16),
        torch.rand(3, 36),
        temporal_scale=0.01,
    )

    out["action_temporal_delta"].sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]

    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert max(gradient.abs().max().item() for gradient in gradients) < 100.0
