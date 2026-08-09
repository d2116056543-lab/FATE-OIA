import torch

from fate_oia.utils.pact_pair_queue import PACTBalancedPairQueue


def test_pair_queue_can_cover_all_21_reasons():
    queue = PACTBalancedPairQueue(21, positive_capacity=4, negative_capacity=4)
    logits = torch.zeros(2, 21)
    target = torch.stack((torch.ones(21), torch.zeros(21)))
    queue.enqueue(logits, target, update=0, counter_priority=torch.ones_like(target))
    _, stats = queue.pairs(update=0, device=torch.device("cpu"))
    assert stats["labels_with_pairs"] == 21
