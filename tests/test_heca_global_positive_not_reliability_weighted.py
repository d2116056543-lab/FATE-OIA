import torch

from fate_oia.losses.meter_reason_losses import build_reason_supervision, robust_reason_asl


def test_observed_positive_weight_is_exactly_one() -> None:
    target = torch.tensor([[1.0, 0.0]])
    supervision = build_reason_supervision(target, torch.tensor([[0.0, 0.2]]))
    assert supervision.positive_weight[0, 0] == 1
    low = robust_reason_asl(torch.zeros(1, 2), supervision)
    supervision2 = build_reason_supervision(target, torch.tensor([[1.0, 0.2]]))
    torch.testing.assert_close(low, robust_reason_asl(torch.zeros(1, 2), supervision2))

