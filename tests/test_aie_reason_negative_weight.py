import torch

from fate_oia.losses.aie_losses import evidence_censored_reason_asl_loss, reason_negative_weight, reason_ranking_loss


def test_reason_negative_weight_preserves_positive_and_bounds_zero():
    target = torch.tensor([[1.0, 0.0, 0.0]])
    counter = torch.tensor([[0.2, 0.0, 1.0]], requires_grad=True)
    weight = reason_negative_weight(target, counter)
    torch.testing.assert_close(weight, torch.tensor([[1.0, 0.25, 1.0]]))
    loss = evidence_censored_reason_asl_loss(torch.zeros_like(target, requires_grad=True), target, counter)
    loss.backward()
    assert counter.grad is None


def test_reason_ranking_uses_evidence_censored_negative_weights():
    logits = torch.tensor([[0.0, 2.0, 0.5]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    negative_weight = torch.tensor([[1.0, 0.25, 1.0]], requires_grad=True)
    loss = reason_ranking_loss(logits, target, negative_weight=negative_weight, margin=0.2)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert negative_weight.grad is None


def test_reason_ranking_penalizes_bad_per_label_order_across_samples():
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bad = torch.tensor([[0.0, -1.0], [1.0, 2.0]])
    good = torch.tensor([[2.0, -1.0], [1.0, 2.0]])
    weight = reason_negative_weight(target, torch.ones_like(target))
    assert reason_ranking_loss(bad, target, weight) > reason_ranking_loss(good, target, weight)


def test_reason_ranking_uses_reference_positive_for_rare_label():
    target = torch.zeros(2, 2)
    reference_target = torch.tensor([[1.0, 0.0]])
    reference_logits = torch.tensor([[1.0, 0.0]])
    bad = reason_ranking_loss(
        torch.tensor([[2.0, 0.0], [1.5, 0.0]]),
        target,
        reference_logits=reference_logits,
        reference_target=reference_target,
    )
    good = reason_ranking_loss(
        torch.tensor([[-1.0, 0.0], [-2.0, 0.0]]),
        target,
        reference_logits=reference_logits,
        reference_target=reference_target,
    )
    assert good < bad


def test_multi_sample_ranking_does_not_fall_back_to_cross_label_ordering():
    logits = torch.tensor([[3.0, -3.0], [2.0, -2.0]], requires_grad=True)
    target = torch.zeros_like(logits)
    loss = reason_ranking_loss(logits, target)
    torch.testing.assert_close(loss, logits.sum() * 0)
