import torch

from fate_oia.losses.meter_reason_losses import noisy_zero_trust


def test_missing_positive_trust_is_product_and_never_makes_hard_positive() -> None:
    trust, weight = noisy_zero_trust(
        torch.tensor([[0.8]]),
        torch.tensor([[0.5]]),
        torch.tensor([[0.5]]),
        torch.tensor([[0.5]]),
    )
    torch.testing.assert_close(trust, torch.tensor([[0.1]]))
    torch.testing.assert_close(weight, torch.tensor([[0.9]]))
    assert 0.10 <= weight.item() <= 1.0

