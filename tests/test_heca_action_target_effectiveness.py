from __future__ import annotations

import torch

from fate_oia.losses.meter_action_losses import action_target_effectiveness_loss
from fate_oia.losses.meter_counterfactual_losses import identity_corruption_loss


def _inputs(*, grounded: bool = True) -> dict[str, torch.Tensor]:
    return {
        "visual_logits": torch.tensor([[0.04, -0.04], [1.50, -1.50]], requires_grad=True),
        "action_delta": torch.tensor([[0.01, -0.01], [0.01, -0.01]], requires_grad=True),
        "control_delta": torch.zeros(2, 2),
        "target": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "factor_weights": torch.full((2, 2, 3), 1.0 / 3.0),
        "reliability": torch.ones(2, 3),
        "ownership": torch.ones(3),
        "groundable": torch.ones(3) if grounded else torch.zeros(3),
    }


def test_target_effectiveness_uses_only_hard_grounded_action_routes() -> None:
    value = _inputs()
    loss, stats = action_target_effectiveness_loss(
        value["visual_logits"],
        value["action_delta"],
        value["control_delta"],
        value["target"],
        value["factor_weights"],
        value["reliability"],
        value["ownership"],
        value["groundable"],
        margin=0.05,
        hard_visual_margin=0.20,
        min_support=0.10,
    )

    # Only the first row is near the visual decision boundary. Its positive
    # and negative action routes both receive direct selected-vs-control loss.
    assert torch.isclose(stats["active_fraction"], torch.tensor(0.5))
    assert torch.isclose(stats["active_count"], torch.tensor(2.0))
    loss.backward()
    assert value["action_delta"].grad is not None
    assert value["action_delta"].grad[0].abs().sum() > 0
    assert value["action_delta"].grad[1].eq(0).all()
    # The visual anchor is intentionally detached: this objective trains
    # evidence transport, never reoptimises the visual action branch.
    assert value["visual_logits"].grad is None


def test_target_effectiveness_requires_grounded_reliable_support() -> None:
    value = _inputs(grounded=False)
    loss, stats = action_target_effectiveness_loss(
        value["visual_logits"],
        value["action_delta"],
        value["control_delta"],
        value["target"],
        value["factor_weights"],
        value["reliability"],
        value["ownership"],
        value["groundable"],
    )

    assert stats["active_count"].item() == 0.0
    assert loss.item() == 0.0
    loss.backward()
    assert value["action_delta"].grad is not None
    assert value["action_delta"].grad.eq(0).all()


def test_target_effectiveness_prefers_clean_target_aligned_credit() -> None:
    value = _inputs()
    good_loss, _ = action_target_effectiveness_loss(
        value["visual_logits"],
        torch.tensor([[0.04, -0.04], [0.0, 0.0]]),
        torch.zeros(2, 2),
        value["target"],
        value["factor_weights"],
        value["reliability"],
        value["ownership"],
        value["groundable"],
        margin=0.05,
        hard_visual_margin=0.20,
        min_support=0.10,
    )
    bad_loss, _ = action_target_effectiveness_loss(
        value["visual_logits"],
        torch.tensor([[-0.04, 0.04], [0.0, 0.0]]),
        torch.zeros(2, 2),
        value["target"],
        value["factor_weights"],
        value["reliability"],
        value["ownership"],
        value["groundable"],
        margin=0.05,
        hard_visual_margin=0.20,
        min_support=0.10,
    )

    assert good_loss < bad_loss


def test_target_effectiveness_rejects_wrong_direction_even_if_control_is_worse() -> None:
    """Relative superiority alone must not reward a wrong-sign action delta."""
    value = _inputs()
    # For the positive action, -0.01 is less harmful than the -0.04 control,
    # so a relative-only objective would call it better. Directional safety
    # must still prefer the positive selected delta.
    wrong_sign_loss, _ = action_target_effectiveness_loss(
        value["visual_logits"],
        torch.tensor([[-0.01, 0.01], [0.0, 0.0]]),
        torch.tensor([[-0.04, 0.04], [0.0, 0.0]]),
        value["target"],
        value["factor_weights"],
        value["reliability"],
        value["ownership"],
        value["groundable"],
        margin=0.01,
        hard_visual_margin=0.20,
        min_support=0.10,
    )
    aligned_loss, _ = action_target_effectiveness_loss(
        value["visual_logits"],
        torch.tensor([[0.01, -0.01], [0.0, 0.0]]),
        torch.tensor([[-0.04, 0.04], [0.0, 0.0]]),
        value["target"],
        value["factor_weights"],
        value["reliability"],
        value["ownership"],
        value["groundable"],
        margin=0.01,
        hard_visual_margin=0.20,
        min_support=0.10,
    )

    assert aligned_loss < wrong_sign_loss


def test_target_effectiveness_can_use_a_visual_scale_relative_margin() -> None:
    value = _inputs()
    _, stats = action_target_effectiveness_loss(
        value["visual_logits"],
        value["action_delta"],
        value["control_delta"],
        value["target"],
        value["factor_weights"],
        value["reliability"],
        value["ownership"],
        value["groundable"],
        relative_margin_fraction=0.05,
        min_margin=0.002,
        max_margin=0.02,
    )

    assert 0.002 <= stats["required_margin_mean"].item() <= 0.02


def test_identity_corruption_accepts_actual_action_route_deltas() -> None:
    target = torch.tensor([[1.0, 0.0]])
    clean = torch.tensor([[0.04, -0.04]])
    control = torch.zeros_like(clean)

    aligned = identity_corruption_loss(clean, control, target, margin=0.05)
    reversed_effect = identity_corruption_loss(control, clean, target, margin=0.05)

    assert aligned < reversed_effect
