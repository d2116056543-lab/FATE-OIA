import torch

from fate_oia.losses.meter_reason_losses import cross_view_consistency


def test_view_consistency_decreases_for_logit_or_measurement_mismatch() -> None:
    same = cross_view_consistency(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2))
    changed = cross_view_consistency(torch.zeros(1, 2), torch.ones(1, 2), torch.zeros(1, 2), torch.ones(1, 2))
    assert torch.allclose(same, torch.ones_like(same))
    assert torch.all(changed < same)

