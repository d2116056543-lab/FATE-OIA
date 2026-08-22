import torch

from fate_oia.losses.tida_flow_credit_losses import (
    counterfactual_margin_credit_loss,
    image_fallback_no_harm_loss,
    signed_gt_margin,
    transition_alignment_loss,
)


def test_signed_gt_margin_rewards_correct_binary_direction():
    logits = torch.tensor([[2.0, -3.0]])
    target = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(signed_gt_margin(logits, target), torch.tensor([[2.0, 3.0]]))


def test_counterfactual_credit_is_zero_only_after_real_history_wins():
    target = torch.tensor([[1.0, 0.0]])
    winning_real = torch.tensor([[2.0, -2.0]], requires_grad=True)
    weaker_counterfactual = torch.tensor([[1.0, -1.0]])
    assert counterfactual_margin_credit_loss(winning_real, weaker_counterfactual, target, margin=0.2).item() == 0

    losing_real = torch.tensor([[0.5, -0.5]], requires_grad=True)
    stronger_counterfactual = torch.tensor([[1.0, -1.0]])
    loss = counterfactual_margin_credit_loss(losing_real, stronger_counterfactual, target, margin=0.2)
    assert loss.item() > 0
    loss.backward()
    assert losing_real.grad is not None and torch.isfinite(losing_real.grad).all()


def test_reason_credit_uses_pu_negative_weights_without_downweighting_positives():
    real = torch.tensor([[0.0, 0.0]])
    counterfactual = torch.tensor([[1.0, 1.0]])
    target = torch.tensor([[1.0, 0.0]])
    weights = torch.tensor([[1.0, 0.2]])
    loss = counterfactual_margin_credit_loss(real, counterfactual, target, sample_weight=weights, margin=0.0)
    # The positive keeps unit weight; the inactive unknown negative remains in
    # the PU normalization with weight 0.2.
    torch.testing.assert_close(loss, torch.tensor(1.0 / 1.2))


def test_image_no_harm_penalizes_only_margin_regression():
    target = torch.tensor([[1.0, 0.0]])
    image = torch.tensor([[1.0, -1.0]])
    better_video = torch.tensor([[2.0, -2.0]])
    worse_video = torch.tensor([[0.0, 0.0]])
    assert image_fallback_no_harm_loss(image, better_video, target).item() == 0
    assert image_fallback_no_harm_loss(image, worse_video, target).item() > 0


def test_transition_alignment_trains_projection_without_moving_target():
    transition = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.randn(2, 3, 4, requires_grad=True)
    loss = transition_alignment_loss(transition, target, torch.ones(2, 3))
    loss.backward()
    assert transition.grad is not None and transition.grad.abs().sum() > 0
    assert target.grad is None
