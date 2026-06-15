import torch

from fate_oia.losses.acpr_threshold_losses import (
    action_cardinality_loss,
    calalign_loss_bundle,
    predicted_positive_rate_loss,
    soft_f1_loss,
    threshold_teacher_loss,
)


def test_threshold_losses_are_finite_and_backprop_to_deploy_logits():
    logits = torch.randn(8, 25, requires_grad=True)
    targets = (torch.rand(8, 25) > 0.7).float()

    loss = soft_f1_loss(logits, targets)
    loss = loss + predicted_positive_rate_loss(logits, targets.mean(0))
    loss = loss + threshold_teacher_loss(torch.zeros(25, requires_grad=True), torch.ones(25))
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_calalign_bundle_contains_required_terms():
    action_logits = torch.randn(6, 4, requires_grad=True)
    reason_logits = torch.randn(6, 21, requires_grad=True)
    action_targets = (torch.rand(6, 4) > 0.5).float()
    reason_targets = (torch.rand(6, 21) > 0.8).float()
    threshold = torch.zeros(25, requires_grad=True)
    teacher = torch.ones(25) * -0.2
    prior = torch.zeros(25)
    target_rate = torch.cat([action_targets.mean(0), reason_targets.mean(0)])

    losses = calalign_loss_bundle(
        action_logits,
        reason_logits,
        action_targets,
        reason_targets,
        threshold,
        teacher,
        prior,
        target_rate,
    )

    for key in [
        "loss_threshold_soft_f1_action",
        "loss_threshold_soft_f1_reason",
        "loss_threshold_rate",
        "loss_action_cardinality",
        "loss_threshold_teacher",
        "loss_threshold_prior",
        "loss_threshold_range",
        "total",
    ]:
        assert key in losses
        assert torch.isfinite(losses[key])
    losses["total"].backward()
    assert action_logits.grad is not None
    assert reason_logits.grad is not None


def test_action_cardinality_loss_is_finite():
    logits = torch.randn(4, 4)
    targets = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.float32)
    assert torch.isfinite(action_cardinality_loss(logits, targets))
