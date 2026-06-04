import torch

from fate_oia.losses.ceai_losses import build_pair_seed_targets


def test_positive_positive_outside_weak_group_is_unknown_not_positive():
    action = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    reason = torch.zeros(1, 21)
    reason[:, 5] = 1.0
    target, mask, weight = build_pair_seed_targets(action, reason, weak_groups={0: [0, 1]})
    assert target[0, 0, 5].item() == 0.0
    assert mask[0, 0, 5].item() == 0.0
    assert weight[0, 0, 5].item() == 0.0


def test_positive_positive_inside_weak_group_is_low_weight_positive():
    action = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    reason = torch.zeros(1, 21)
    reason[:, 5] = 1.0
    q_ar = torch.full((1, 4, 21), 0.5)
    target, mask, weight = build_pair_seed_targets(action, reason, q_ar=q_ar, weak_groups={0: [5]})
    assert target[0, 0, 5].item() == 1.0
    assert mask[0, 0, 5].item() == 1.0
    assert 0.0 < weight[0, 0, 5].item() < 0.2


def test_negative_action_or_reason_pairs_are_masked_negatives():
    action = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    reason = torch.zeros(1, 21)
    reason[:, 5] = 1.0
    target, mask, weight = build_pair_seed_targets(action, reason, weak_groups={1: [5]})
    assert target[0, 0, 5].item() == 0.0
    assert mask[0, 0, 5].item() == 1.0
    assert weight[0, 0, 5].item() > 0.0
