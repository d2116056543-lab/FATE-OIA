from __future__ import annotations

import importlib

import torch


def test_reason_anchors_are_positive_only_and_drive_factor_gradients() -> None:
    module = importlib.import_module("fate_oia.datasets.mosaic_icdor_factor_supervision")
    reason = torch.zeros(2, 21)
    reason[0, 3] = 1.0
    observations = {
        "presence_target": torch.tensor([[0.0], [1.0]]),
        "presence_known_mask": torch.tensor([[0.0], [1.0]]),
        "geometry_known_mask": torch.tensor([[0.0], [1.0]]),
        "weak_negative_mask": torch.tensor([[0.0], [0.0]]),
    }
    result = module.build_factor_supervision(
        observations,
        reason,
        ({"name": "traffic_control", "positive_reason_anchors": [3], "grounding_source": "box2d"},),
        split="train_core",
    )
    assert result["positive_anchor_mask"].tolist() == [[True], [False]]
    assert result["positive_anchor_weight"][0, 0].item() == 0.35
    assert not result["reliable_negative_mask"][0, 0]
    assert result["geometry_positive_mask"][1, 0]
    assert result["positive_anchor_weight"][1, 0].item() == 1.0

    logits = torch.zeros(2, 1, requires_grad=True)
    loss = module.factor_positive_anchor_loss(logits, result)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0.0


def test_reason_anchors_are_rejected_outside_train_core_and_train_audit() -> None:
    module = importlib.import_module("fate_oia.datasets.mosaic_icdor_factor_supervision")
    observations = {key: torch.zeros(1, 1) for key in (
        "presence_target", "presence_known_mask", "geometry_known_mask", "weak_negative_mask"
    )}
    reason = torch.zeros(1, 21)
    reason[0, 3] = 1.0
    result = module.build_factor_supervision(
        observations, reason, ({"name": "f", "positive_reason_anchors": [3], "grounding_source": "image_only"},), split="test"
    )
    assert not result["positive_anchor_mask"].any()


def test_visual_factor_supervision_can_disable_reason_label_anchors() -> None:
    """CREDO factor measurements must not learn visual positives from reasons."""
    module = importlib.import_module("fate_oia.datasets.mosaic_icdor_factor_supervision")
    observations = {key: torch.zeros(1, 1) for key in (
        "presence_target", "presence_known_mask", "geometry_known_mask", "weak_negative_mask"
    )}
    reason = torch.zeros(1, 21)
    reason[0, 3] = 1.0

    result = module.build_factor_supervision(
        observations,
        reason,
        ({"name": "traffic_control", "positive_reason_anchors": [3], "grounding_source": "box2d"},),
        split="train_core",
        allow_reason_anchors=False,
    )

    assert not result["positive_anchor_mask"].any()
    assert not result["supervision_mask"].any()

