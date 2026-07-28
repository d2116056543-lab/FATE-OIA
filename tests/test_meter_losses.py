import torch

from fate_oia.losses.meter_action_losses import (
    meter_action_loss,
    meter_action_loss_per_sample,
)
from fate_oia.losses.meter_reason_losses import meter_reason_loss


def test_meter_action_loss_supervises_final_visual_semantic_and_selector() -> None:
    logits = {name: torch.randn(3, 4, requires_grad=True) for name in ("action_logits_final", "action_logits_visual", "action_logits_semantic", "action_logits_peer")}
    logits["action_selector"] = torch.sigmoid(torch.randn(3, 4, requires_grad=True))
    target = torch.randint(0, 2, (3, 4)).float()
    result = meter_action_loss(logits, target)
    assert {"final", "visual", "semantic", "two_way", "soft_f1", "cardinality", "selector_regret", "total"} <= result.keys()
    result["total"].backward()
    assert logits["action_logits_final"].grad is not None


def test_meter_action_loss_per_sample_is_finite_and_differentiable() -> None:
    logits = {
        name: torch.randn(6, 4, requires_grad=True)
        for name in (
            "action_logits_final",
            "action_logits_visual",
            "action_logits_semantic",
            "action_logits_peer",
        )
    }
    target = torch.randint(0, 2, (6, 4)).float()

    formal = meter_action_loss(logits, target)["total"]
    losses = meter_action_loss_per_sample(logits, target)

    assert losses.shape == (6,)
    assert torch.isfinite(losses).all()
    torch.testing.assert_close(losses.mean(), formal)
    formal_gradients = torch.autograd.grad(
        formal, tuple(logits.values()), retain_graph=True
    )
    per_sample_gradients = torch.autograd.grad(
        losses.mean(), tuple(logits.values())
    )
    for formal_gradient, per_sample_gradient in zip(
        formal_gradients, per_sample_gradients
    ):
        torch.testing.assert_close(per_sample_gradient, formal_gradient)


def test_meter_reason_loss_is_weighted_and_has_direct_private_view_terms() -> None:
    logits = {name: torch.randn(2, 21, requires_grad=True) for name in ("reason_logits_final", "reason_logits_global", "reason_logits_local")}
    logits["reason_annotation_delta"] = torch.randn(2, 21, requires_grad=True)
    target = torch.randint(0, 2, (2, 21)).float()
    confidence = torch.rand(2, 21)
    result = meter_reason_loss(logits, target, confidence)
    assert {"final", "global", "local", "rank", "soft_f1", "annotation_delta", "total"} <= result.keys()
    result["total"].backward()
    assert logits["reason_logits_global"].grad is not None
    assert logits["reason_logits_local"].grad is not None
