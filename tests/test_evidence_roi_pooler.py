from __future__ import annotations

import torch

from fate_oia.models.evidence_roi_pooler import EvidenceROIPooler, box2d_to_patch_weights, poly2d_to_patch_weights


def test_box2d_to_patch_weights_hits_overlapping_patches() -> None:
    weights = box2d_to_patch_weights({"x1": 0, "y1": 0, "x2": 64, "y2": 64}, image_size=(128, 128), patch_grid=(4, 4))
    assert weights.shape == (16,)
    assert float(weights.sum()) > 0.0


def test_poly2d_to_patch_weights_handles_vertices() -> None:
    poly = {"vertices": [[0, 0], [127, 0], [127, 127], [0, 127]]}
    weights = poly2d_to_patch_weights(poly, image_size=(128, 128), patch_grid=(4, 4))
    assert weights.shape == (16,)
    assert float(weights.sum()) > 8.0


def test_evidence_roi_pooler_pools_tokens_from_mask_weights() -> None:
    tokens = torch.randn(2, 16, 32)
    weights = torch.zeros(2, 3, 16)
    weights[:, 0, :4] = 1.0
    weights[:, 1, 4:8] = 1.0
    pooler = EvidenceROIPooler(dim=32)
    out = pooler(tokens, weights)
    assert out["evidence_tokens"].shape == (2, 3, 32)
    assert out["valid_mask"].shape == (2, 3)
    assert out["valid_mask"][:, 0].all()
