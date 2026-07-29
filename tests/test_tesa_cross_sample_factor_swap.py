import torch

from fate_oia.engine.tesa_diagnostics import cross_sample_same_factor_swap


def test_cross_sample_swap_preserves_factor_index() -> None:
    token = torch.arange(3 * 21 * 2).view(3, 21, 2)
    swapped = cross_sample_same_factor_swap(token)
    assert torch.equal(swapped[0, 7], token[-1, 7])
    assert torch.equal(swapped[:, 7], torch.roll(token[:, 7], 1, 0))
