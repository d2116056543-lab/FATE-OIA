import torch
from fate_oia.losses.acpr_pmt_losses import predicate_pair_weights


def test_weak_predicate_pairs_downweighted_or_rejected():
    diff = torch.tensor([0.01, 0.08])
    weights, stats = predicate_pair_weights(diff, threshold=0.05)
    assert weights[0] <= 0.25
    assert weights[1] == 1.0
    assert stats["predicate_filtered_pair_count"] == 1
    assert stats["weak_predicate_pair_count"] == 1
