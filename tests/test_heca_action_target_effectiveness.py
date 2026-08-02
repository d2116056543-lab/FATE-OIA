from __future__ import annotations

import torch

from fate_oia.losses.meter_action_losses import action_target_effectiveness_loss


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
