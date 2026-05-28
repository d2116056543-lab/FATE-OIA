from __future__ import annotations

import torch

from fate_oia.models.multilayer_dino_features import MultiLayerDINOFeatureFusion


def test_multilayer_fusion_preserves_token_shape_and_normalizes_weights() -> None:
    layers = [torch.randn(2, 17, 32), torch.randn(2, 17, 32), torch.randn(2, 17, 32)]
    fusion = MultiLayerDINOFeatureFusion(dim=32, num_layers=3, dropout=0.0)
    out = fusion(layers)
    assert out["tokens"].shape == (2, 17, 32)
    assert out["layer_weights"].shape == (3,)
    assert torch.allclose(out["layer_weights"].sum(), torch.tensor(1.0), atol=1e-6)


def test_multilayer_fusion_rejects_shape_mismatch() -> None:
    fusion = MultiLayerDINOFeatureFusion(dim=32, num_layers=2)
    bad_layers = [torch.randn(2, 17, 32), torch.randn(2, 16, 32)]
    try:
        fusion(bad_layers)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch to raise ValueError")
