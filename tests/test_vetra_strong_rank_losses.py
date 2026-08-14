import torch

from fate_oia.losses.vetra_strong_rank_losses import (
    action_pairwise_ap_loss,
    action_smooth_ap_loss,
    base_margin_trust_loss,
)


def test_pairwise_loss_rewards_better_positive_negative_ordering():
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    bad = torch.tensor([[-1.0], [0.0], [0.5], [1.0]])
    good = torch.tensor([[1.0], [0.5], [0.0], [-1.0]])
    assert action_pairwise_ap_loss(good, target) < action_pairwise_ap_loss(bad, target)
    assert action_smooth_ap_loss(good, target) < action_smooth_ap_loss(bad, target)


def test_trust_loss_is_zero_for_preserved_or_improved_margins():
    target = torch.tensor([[1.0], [0.0]])
    base = torch.tensor([[0.5], [-0.5]])
    improved = torch.tensor([[0.8], [-0.8]])
    degraded = torch.tensor([[0.2], [-0.2]])
    assert base_margin_trust_loss(improved, base, target) == 0
    assert base_margin_trust_loss(degraded, base, target) > 0
