import torch
from fate_oia.losses.acpr_ntmcal_losses import schedule_weights, ntmcal_reason_pu_loss, native_predicate_measurement_loss

def test_loss_schedule_and_pu_unknown():
    assert schedule_weights(0)["pair"] == 0
    assert schedule_weights(7)["pair"] > 0
    logits = torch.zeros(2,21, requires_grad=True)
    pu = {"positive_mask": torch.zeros(2,21), "soft_negative_weight": torch.zeros(2,21), "hard_negative_mask": torch.zeros(2,21)}
    loss = ntmcal_reason_pu_loss(logits, torch.zeros(2,21), pu, 0)
    assert loss.item() == 0

def test_reason_pu_loss_detaches_reliability_weights():
    logits = torch.zeros(2, 3, requires_grad=True)
    reason_targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    soft_neg = torch.full((2, 3), 0.4, requires_grad=True)
    hard_neg = torch.zeros(2, 3, requires_grad=True)
    pu_state = {
        "positive_mask": reason_targets,
        "soft_negative_weight": soft_neg,
        "hard_negative_mask": hard_neg,
    }
    loss = ntmcal_reason_pu_loss(logits, reason_targets, pu_state, epoch=7)
    loss.backward()
    assert logits.grad is not None
    assert soft_neg.grad is None
    assert hard_neg.grad is None


def test_predicate_measurement_loss_pushes_observed_rho_up():
    q = torch.full((2, 4), 0.5, requires_grad=True)
    rho = torch.full((2, 4), 0.05, requires_grad=True)
    observations = {
        "obs_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "obs_value": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "obs_soft_negative": torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0]]),
    }
    loss = native_predicate_measurement_loss(q, rho, observations, epoch=7)
    loss.backward()
    observed = (observations["obs_mask"] + observations["obs_soft_negative"]) > 0
    assert rho.grad[observed].mean() < 0.0


