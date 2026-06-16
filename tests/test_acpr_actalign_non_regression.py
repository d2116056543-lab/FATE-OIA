import torch

from fate_oia.models.acpr_action_utility import ACPRActionUtility
from fate_oia.losses.acpr_action_utility_losses import action_utility_nonregression_loss


def test_utility_nonregression_zero_when_equal_or_better():
    fallback = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0]])
    equal = fallback.clone().requires_grad_(True)
    assert action_utility_nonregression_loss(equal, fallback, targets).item() == 0.0
    better = torch.tensor([[4.0, -4.0]], requires_grad=True)
    assert action_utility_nonregression_loss(better, fallback, targets).item() == 0.0
    worse = torch.tensor([[-4.0, 4.0]], requires_grad=True)
    loss = action_utility_nonregression_loss(worse, fallback, targets)
    assert loss.item() > 0
    loss.backward()
    assert worse.grad is not None
