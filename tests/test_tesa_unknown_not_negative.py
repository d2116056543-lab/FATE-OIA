import torch

from fate_oia.losses.meter_grounding_losses import conditional_state_ce


def test_unknown_state_has_zero_gradient() -> None:
    logits = torch.randn(1, 2, 3, requires_grad=True)
    target = torch.tensor([[-1, 1]])
    loss = conditional_state_ce(logits, target, torch.tensor([[False, True]]), torch.ones(1, 2))
    loss.backward()
    assert logits.grad[0, 0].eq(0).all()
    assert logits.grad[0, 1].abs().sum() > 0
