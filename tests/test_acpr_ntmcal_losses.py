import torch
from fate_oia.losses.acpr_ntmcal_losses import schedule_weights, ntmcal_reason_pu_loss

def test_loss_schedule_and_pu_unknown():
    assert schedule_weights(0)["pair"] == 0
    assert schedule_weights(7)["pair"] > 0
    logits = torch.zeros(2,21, requires_grad=True)
    pu = {"positive_mask": torch.zeros(2,21), "soft_negative_weight": torch.zeros(2,21), "hard_negative_mask": torch.zeros(2,21)}
    loss = ntmcal_reason_pu_loss(logits, torch.zeros(2,21), pu, 0)
    assert loss.item() == 0
