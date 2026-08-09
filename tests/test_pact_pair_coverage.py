import torch

from fate_oia.engine.train_pact_oia_probe import accumulate_reason_coverage
from fate_oia.utils.pact_pair_queue import PACTBalancedPairQueue


def test_pair_queue_can_cover_all_21_reasons():
    queue = PACTBalancedPairQueue(21, positive_capacity=4, negative_capacity=4)
    logits = torch.zeros(2, 21)
    target = torch.stack((torch.ones(21), torch.zeros(21)))
    queue.enqueue(logits, target, update=0, counter_priority=torch.ones_like(target))
    _, stats = queue.pairs(update=0, device=torch.device("cpu"))
    assert stats["labels_with_pairs"] == 21


def test_epoch_reason_coverage_is_union_not_last_batch_snapshot():
    aggregate = None
    aggregate = accumulate_reason_coverage(aggregate, {
        "positive_count": [1, 0, 0], "negative_count": [1, 1, 0], "pair_count": [1, 0, 0],
        "labels_with_pairs": 1,
    })
    aggregate = accumulate_reason_coverage(aggregate, {
        "positive_count": [0, 1, 1], "negative_count": [0, 1, 1], "pair_count": [0, 1, 1],
        "labels_with_pairs": 2,
    })
    assert aggregate["pair_count"] == [1, 1, 1]
    assert aggregate["labels_with_pairs"] == 3
    assert aggregate["latest_labels_with_pairs"] == 2
