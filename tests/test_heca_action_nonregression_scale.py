import torch

from fate_oia.losses.meter_action_losses import (
    action_delta_pairwise_ranking_loss,
    action_nonregression_loss,
)


def test_nonregression_protects_relatively_confident_correct_visual_margins_at_small_scale() -> None:
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    visual = torch.tensor([[0.20, -0.18, 0.12, -0.10]])
    delta = -(target * 2.0 - 1.0) * 0.20
    loss = action_nonregression_loss(
        visual,
        delta,
        target,
        min_margin=0.05,
        confidence_quantile=0.50,
    )
    assert loss.item() >= 0.20


def test_nonregression_skips_visual_predictions_with_no_correct_margin() -> None:
    target = torch.tensor([[1.0, 0.0]])
    visual = torch.tensor([[-0.2, 0.2]])
    assert action_nonregression_loss(visual, torch.zeros_like(visual), target).eq(0)


def test_nonregression_updates_delta_but_not_visual_anchor() -> None:
    target = torch.tensor([[1.0, 0.0]])
    visual = torch.tensor([[0.2, -0.2]], requires_grad=True)
    delta = torch.tensor([[-0.2, 0.2]], requires_grad=True)
    loss = action_nonregression_loss(visual, delta, target, confidence_quantile=0.0)
    loss.backward()
    assert visual.grad is None or visual.grad.eq(0).all()
    assert delta.grad is not None and delta.grad.abs().sum() > 0


def test_nonregression_guards_every_correct_prediction_from_a_sign_flip() -> None:
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    visual = torch.tensor([[0.11, -0.13, 0.16, -0.19]], requires_grad=True)
    # These margins are correct but deliberately below the high-confidence
    # quantile. The residual must still be penalised if it flips them.
    delta = torch.tensor([[-0.20, 0.20, -0.20, 0.20]], requires_grad=True)

    loss = action_nonregression_loss(
        visual, delta, target, confidence_quantile=1.0
    )
    loss.backward()

    assert loss.item() > 0.0
    assert visual.grad is None or visual.grad.eq(0).all()
    assert delta.grad is not None and delta.grad.abs().sum() > 0


def test_credit_rank_focuses_visual_misrankings_without_moving_easy_pairs() -> None:
    visual = torch.tensor([[1.0], [-0.10], [-1.0], [0.10]])
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    helpful = torch.tensor([[0.0], [0.30], [0.0], [-0.30]], requires_grad=True)
    harmful = torch.tensor([[0.0], [-0.20], [0.0], [0.20]], requires_grad=True)

    helpful_loss = action_delta_pairwise_ranking_loss(
        helpful, target, visual_logits=visual
    )
    harmful_loss = action_delta_pairwise_ranking_loss(
        harmful, target, visual_logits=visual
    )
    harmful_loss.backward()

    assert helpful_loss.item() < harmful_loss.item()
    assert harmful.grad is not None and harmful.grad.abs().sum() > 0


def test_nonregression_guards_correct_near_boundary_predictions() -> None:
    target = torch.tensor([[1.0, 0.0]])
    visual = torch.tensor([[0.01, -0.01]], requires_grad=True)
    # Both entries are correct under the fixed zero threshold, but the delta
    # would flip them.  The high-confidence guard alone would skip this row.
    delta = torch.tensor([[-0.03, 0.03]], requires_grad=True)

    loss = action_nonregression_loss(
        visual,
        delta,
        target,
        min_margin=0.05,
        boundary_margin=0.02,
    )
    loss.backward()

    assert loss > 0
    assert visual.grad is None or visual.grad.eq(0).all()
    assert delta.grad is not None and delta.grad.abs().sum() > 0
