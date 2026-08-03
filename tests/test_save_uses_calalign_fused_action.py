from __future__ import annotations

import torch

from fate_oia.models.save_oia_model import SAVEOIAModel


def test_formal_save_base_uses_fused_calalign_action_not_visual_auxiliary() -> None:
    model = SAVEOIAModel(use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)
    field = model.encode_images(images)
    foundation = model.foundation.decode_foundation(field)
    foundation["action_logits_visual_base"] = torch.full_like(
        foundation["action_logits_calalign"], 17.0
    )

    original_decode = model.foundation.decode_foundation
    model.foundation.decode_foundation = lambda _: foundation  # type: ignore[method-assign]
    try:
        output = model.decode_from_field(field, progress=0.0)
    finally:
        model.foundation.decode_foundation = original_decode  # type: ignore[method-assign]

    torch.testing.assert_close(
        output["action_logits_base"], foundation["action_logits_calalign"]
    )
    torch.testing.assert_close(
        output["action_logits_final"], foundation["action_logits_calalign"]
    )
    assert not torch.equal(output["action_logits_base"], foundation["action_logits_visual_base"])
