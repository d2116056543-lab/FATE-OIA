import torch

from fate_oia.losses.meter_action_losses import action_nonregression_loss


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
    assert torch.isclose(loss, torch.tensor(0.20), atol=1e-6)


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
