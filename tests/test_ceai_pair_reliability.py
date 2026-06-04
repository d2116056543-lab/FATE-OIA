import torch

from fate_oia.models.ceai_pair_reliability import PairReliabilityHead, build_pair_seed_targets


def test_pair_reliability_shapes_and_seed_gate():
    head = PairReliabilityHead(dim=16, action_dim=4, reason_dim=21, reason_group_count=6)
    action = torch.randn(2, 4, 16)
    reason = torch.randn(2, 21, 16)
    context = torch.randn(2, 4, 6, 16)
    out = head(action, reason, context)
    assert out["pair_support"].shape == (2, 4, 21)
    assert out["pair_reliability"].shape == (2, 4, 21)
    assert out["reason_reliability"].shape == (2, 21)
    labels_a = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 0]], dtype=torch.float32)
    labels_r = torch.zeros(2, 21)
    labels_r[:, [5, 9]] = 1
    target, weight = build_pair_seed_targets(labels_a, labels_r, out["pair_reliability"])
    assert target.shape == (2, 4, 21)
    assert weight.shape == (2, 4, 21)
    assert weight.max() <= 1.0 and weight.min() >= 0.0
