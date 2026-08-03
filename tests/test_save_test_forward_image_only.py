from __future__ import annotations

import pytest
import torch

from fate_oia.models.save_oia_model import SAVEOIAModel


def test_save_test_forward_is_image_only_and_does_not_require_targets() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    output = model.forward_test(torch.randn(1, 3, 360, 640))

    assert model.encode_call_count == 1
    assert output["test_forward_image_only"] is True
    assert output["utility_teacher_plan"] is None
    assert output["counterfactual_teacher"] is None
    assert output["action_logits_final"].shape == (1, 4)
    assert output["reason_logits_final"].shape == (1, 21)


def test_save_test_forward_rejects_labels_and_counterfactual_inputs() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)
    with pytest.raises(ValueError, match="image-only"):
        model.forward_test(images, action_targets=torch.zeros(1, 4))
    with pytest.raises(ValueError, match="image-only"):
        model.forward_test(images, run_teacher=True)
