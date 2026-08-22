import torch

from fate_oia.losses.tida_flow_credit_losses import (
    conditional_credit_weight,
    conditional_no_harm_weight,
    temporal_utility_calibration_loss,
)


def test_utility_calibration_prefers_high_budget_for_positive_temporal_benefit():
    target = torch.tensor([[1.0, 0.0]])
    counterfactual = torch.tensor([[0.0, 0.0]])
    helpful = torch.tensor([[1.0, -1.0]])
    harmful = torch.tensor([[-1.0, 1.0]])
    high = torch.tensor([[0.8, 0.8]])
    low = torch.tensor([[0.05, 0.05]])
    assert temporal_utility_calibration_loss(high, helpful, counterfactual, target) < temporal_utility_calibration_loss(low, helpful, counterfactual, target)
    assert temporal_utility_calibration_loss(low, harmful, counterfactual, target) < temporal_utility_calibration_loss(high, harmful, counterfactual, target)


def test_utility_target_is_detached_from_paired_logits():
    budget = torch.full((2, 4), 0.2, requires_grad=True)
    real = torch.randn(2, 4, requires_grad=True)
    counterfactual = torch.randn(2, 4, requires_grad=True)
    temporal_utility_calibration_loss(budget, real, counterfactual, torch.ones(2, 4)).backward()
    assert budget.grad is not None and budget.grad.abs().sum() > 0
    assert real.grad is None
    assert counterfactual.grad is None


def test_conditional_credit_and_no_harm_weights_are_complementary():
    need = torch.tensor([[0.0, 0.5, 1.0]])
    credit = conditional_credit_weight(need)
    no_harm = conditional_no_harm_weight(need)
    assert torch.all(credit[:, 1:] > credit[:, :-1])
    assert torch.all(no_harm[:, 1:] < no_harm[:, :-1])
    torch.testing.assert_close(credit + no_harm, torch.full_like(credit, 1.25))
