from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.visual_encoder import InteractVisualEncoder


def test_visual_encoder_consumes_configured_dino_size_grid_anchors_and_chunk() -> None:
    encoder = InteractVisualEncoder(
        use_mock_dino=True,
        dino_input_height=64,
        dino_input_width=96,
        patch_size=8,
        dino_chunk_size=2,
        anchor_frames=(0, 3, 6, 9, 12, 14),
        selected_layers=(3, 7, 11),
    )
    frames = torch.rand(2, 15, 3, 80, 120)

    out = encoder(frames)

    assert out.grid_hw == (8, 12)
    assert out.patch_tokens_by_layer.shape == (2, 6, 3, 96, encoder.dim)
    assert out.stats["dino_input_h"] == 64
    assert out.stats["dino_input_w"] == 96
    assert out.stats["grid_h"] == 8
    assert out.stats["grid_w"] == 12
    assert out.stats["anchor_indices"] == [0, 3, 6, 9, 12, 14]
    assert out.stats["dino_chunk_size"] == 2
    assert out.stats["formal_target_frame_used"] is False
