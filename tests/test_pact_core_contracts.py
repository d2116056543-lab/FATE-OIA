import torch

from fate_oia.losses.pact_rank_losses import action_rank_trust_region
from fate_oia.models.pact_predicate_agreement import PACTPredicateAgreement
from fate_oia.models.pact_shared_readout import licensed_gradient
from fate_oia.utils.pact_pair_queue import PACTBalancedPairQueue


def test_licensed_gradient_preserves_forward_and_scales_backward():
    x = torch.randn(3, requires_grad=True)
    y = licensed_gradient(x, 0.25)
    assert torch.equal(x, y)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, .25))


def test_predicate_agreement_is_bounded_and_falls_back():
    gate = PACTPredicateAgreement(.25)
    visual = torch.tensor([[[.9, .1]]])
    aligned = torch.tensor([[[.9, .1]]])
    opposed = torch.tensor([[[.1, .9]]])
    high = gate(visual, aligned, torch.ones(1, 1))
    low = gate(visual, opposed, torch.ones(1, 1))
    assert 0 <= low["predicate_agreement_strength"].item() < high["predicate_agreement_strength"].item() <= .25


def test_action_rank_repairs_wrong_and_preserves_correct_pairs():
    out = action_rank_trust_region(torch.tensor([2., .2]), torch.tensor([0., .8]),
                                   torch.tensor([1., 0.]), torch.tensor([0., 1.]))
    assert out["preserve_loss"] == 0
    assert out["repair_loss"] > 0
    assert out["new_pair_inversion_rate"] == 0


def test_balanced_queue_can_cover_all_21_labels_with_weak_negatives():
    queue = PACTBalancedPairQueue(21)
    positive = torch.ones(1, 21)
    negative = torch.zeros(1, 21)
    queue.enqueue(torch.ones_like(positive), positive, 0)
    queue.enqueue(torch.zeros_like(negative), negative, 1)
    _, stats = queue.pairs(1, torch.device("cpu"))
    assert stats["labels_with_pairs"] == 21


def test_pair_queue_restore_normalizes_storage_to_cpu():
    queue = PACTBalancedPairQueue(1)
    queue.enqueue(torch.tensor([[1.0], [-1.0]]), torch.tensor([[1.0], [0.0]]), 0)
    restored = PACTBalancedPairQueue(1)
    restored.load_state_dict(queue.state_dict())
    rows, stats = restored.pairs(0, torch.device("cpu"))
    assert stats["labels_with_pairs"] == 1
    assert rows[0][1].device.type == "cpu" and rows[0][2].device.type == "cpu"
