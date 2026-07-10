from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from fate_oia.losses.mosaic_posterior_ranking import (
    action_cross_image_ranking_loss,
    posterior_weighted_reason_ranking_loss,
)
from fate_oia.optim.mosaic_soft_rank_queue import MOSAICSoftRankQueue


def test_soft_rank_queue_is_detached_fixed_capacity_fifo_ring_buffer() -> None:
    queue = MOSAICSoftRankQueue(label_dim=2, capacity=4)
    queue.enqueue(
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        ["a", "b", "c"],
    )
    queue.enqueue(
        torch.tensor([[4.0, 40.0], [5.0, 50.0]], requires_grad=True),
        torch.tensor([[0.7, 0.8], [0.9, 1.0]], requires_grad=True),
        ["d", "e"],
    )
    snapshot = queue.snapshot()
    assert torch.equal(snapshot["logits"], torch.tensor([[2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]))
    assert torch.allclose(snapshot["targets"], torch.tensor([[0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]))
    assert snapshot["logits"].requires_grad is False
    assert snapshot["targets"].requires_grad is False
    assert queue.count == 4
    assert "torch.cat" not in inspect.getsource(MOSAICSoftRankQueue.enqueue)


def test_reason_ranking_uses_exact_soft_posterior_pair_weights() -> None:
    queue = MOSAICSoftRankQueue(label_dim=1, capacity=8)
    queue.enqueue(torch.tensor([[-1.0]]), torch.tensor([[0.25]]), ["history"])
    current_logits = torch.tensor([[0.5]], requires_grad=True)
    current_q = torch.tensor([[0.60]])
    loss, stats = posterior_weighted_reason_ranking_loss(current_logits, current_q, ["current"], queue)
    expected_weight = 0.60 * (1.0 - 0.25)
    expected = expected_weight * F.softplus(torch.tensor(-(0.5 - -1.0))) / expected_weight
    assert torch.allclose(loss, expected)
    assert torch.allclose(stats["pair_weight_sum"], torch.tensor(expected_weight), atol=1e-7)
    loss.backward()
    assert current_logits.grad is not None and current_logits.grad < 0


def test_reason_ranking_never_hard_thresholds_fractional_posterior() -> None:
    queue = MOSAICSoftRankQueue(label_dim=1, capacity=8)
    queue.enqueue(torch.tensor([[0.2]]), torch.tensor([[0.40]]), ["history"])
    loss, stats = posterior_weighted_reason_ranking_loss(
        torch.tensor([[0.1]], requires_grad=True),
        torch.tensor([[0.51]]),
        ["current"],
        queue,
    )
    assert loss > 0
    assert stats["pair_weight_sum"] > 0
    source = inspect.getsource(posterior_weighted_reason_ranking_loss)
    assert "> 0.5" not in source
    assert ".bool()" not in source


def test_action_ranking_is_cross_image_per_action_and_uses_true_labels() -> None:
    queue = MOSAICSoftRankQueue(label_dim=2, capacity=8)
    queue.enqueue(torch.tensor([[-1.0, 2.0]]), torch.tensor([[0.0, 1.0]]), ["negative_for_a0"])
    logits = torch.tensor([[1.0, -2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    loss, stats = action_cross_image_ranking_loss(logits, targets, ["current"], queue)
    expected = F.softplus(torch.tensor(-(1.0 - -1.0)))
    assert torch.allclose(loss, expected)
    assert stats["pair_weight_sum"] == 1
    loss.backward()
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] == 0


def test_ranking_excludes_same_sample_and_empty_queue_keeps_graph() -> None:
    empty = MOSAICSoftRankQueue(label_dim=1, capacity=4)
    logits = torch.tensor([[0.3]], requires_grad=True)
    zero, stats = posterior_weighted_reason_ranking_loss(logits, torch.tensor([[0.8]]), ["same"], empty)
    assert zero == 0 and stats["pair_weight_sum"] == 0
    zero.backward()
    assert logits.grad is not None

    queue = MOSAICSoftRankQueue(label_dim=1, capacity=4)
    queue.enqueue(torch.tensor([[-0.2]]), torch.tensor([[0.0]]), ["same"])
    excluded, stats = posterior_weighted_reason_ranking_loss(
        torch.tensor([[0.5]], requires_grad=True), torch.tensor([[1.0]]), ["same"], queue
    )
    assert excluded == 0 and stats["pair_weight_sum"] == 0
