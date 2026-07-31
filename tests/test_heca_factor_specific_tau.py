import torch

from fate_oia.datasets.meter_typed_targets import compute_factor_observability_tau


def test_factor_specific_tau_uses_group_shrinkage_not_uniform_half() -> None:
    observed = torch.tensor([90.0, 10.0, 50.0])
    valid = torch.tensor([100.0, 100.0, 100.0])
    tau = compute_factor_observability_tau(observed, valid, ["a", "a", "b"], alpha=20)
    assert tau.shape == (3,)
    assert tau[0] > tau[2] > tau[1]
    assert not torch.allclose(tau, torch.full_like(tau, 0.5))
    assert torch.all((tau >= 0.05) & (tau <= 0.95))

