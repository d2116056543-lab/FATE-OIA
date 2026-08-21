import torch

from fate_oia.models.tida_predicate_differential import finite_difference


def test_finite_difference_uses_real_time_spacing():
    times = torch.tensor([[0.0, 1.0, 3.0]])
    values = torch.tensor([[[[0.0]], [[2.0]], [[6.0]]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    velocity, pair_valid = finite_difference(values, times, valid)
    torch.testing.assert_close(velocity.flatten(), torch.tensor([2.0, 2.0]))
    assert pair_valid.all()
