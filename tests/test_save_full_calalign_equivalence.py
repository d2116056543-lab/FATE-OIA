from __future__ import annotations

import torch

from fate_oia.models.save_oia_model import SAVEOIAModel


def test_progress_zero_preserves_full_calalign_foundation_outputs() -> None:
    torch.manual_seed(13)
    model = SAVEOIAModel(use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)

    field = model.encode_images(images)
    expected = model.foundation.decode_foundation(field)
    output = model.decode_from_field(field, progress=0.0)

    for key in (
        "action_logits_calalign",
        "reason_logits_calalign",
        "label_nodes",
        "label_attention",
        "action_fusion_gate",
    ):
        torch.testing.assert_close(output[key].float(), expected[key].float(), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        output["action_logits_final"].float(),
        expected["action_logits_calalign"].float(),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        output["reason_logits_final"].float(),
        expected["reason_logits_calalign"].float(),
        atol=1e-6,
        rtol=0,
    )
