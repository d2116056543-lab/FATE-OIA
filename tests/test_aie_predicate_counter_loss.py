import torch

from fate_oia.losses.aie_losses import predicate_masked_asl_loss


def test_predicate_counter_mask_supplies_negative_supervision_and_unknown_stays_unobserved():
    logits = torch.zeros(1, 3, requires_grad=True)
    loss = predicate_masked_asl_loss(
        logits,
        positive_target=torch.tensor([[1.0, 0.0, 0.0]]),
        positive_mask=torch.tensor([[1.0, 0.0, 0.0]]),
        counter_mask=torch.tensor([[0.0, 1.0, 0.0]]),
        reliability=torch.ones(1, 3),
    )
    loss.backward()
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] > 0
    assert logits.grad[0, 2] == 0
