import torch

from fate_oia.losses.meter_grounding_losses import conditional_state_ce


def test_unknown_state_with_invalid_target_contributes_zero() -> None:
    logits = torch.randn(1, 1, 3)
    loss = conditional_state_ce(logits, torch.tensor([[-1]]), torch.zeros(1, 1), torch.ones(1, 1))
    assert loss == 0

