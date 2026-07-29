import torch

from fate_oia.losses.meter_pu_losses import meter_private_pu_loss, meter_pu_score


def test_inactive_pu_is_exact_zero_and_score_is_exact_product() -> None:
    logits = torch.randn(3, 2, requires_grad=True)
    score = meter_pu_score(torch.full((3, 2), .5), torch.full((3, 2), .4), torch.full((3, 2), .25))
    torch.testing.assert_close(score, torch.full((3, 2), .05))
    loss = meter_private_pu_loss(logits, torch.zeros(3, 2), score, torch.zeros(2))
    loss.backward()
    assert loss == 0
    assert logits.grad.eq(0).all()
