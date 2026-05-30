from __future__ import annotations

import torch

from fate_oia.models.causal_evidence_pooler import CausalEvidencePooler, box_to_patch_mask


def test_evidence_pooler_uses_object_tokens():
    pooler = CausalEvidencePooler(dim=4, per_reason_topk_evidence=2)
    tokens = torch.arange(1 * 17 * 4, dtype=torch.float32).view(1, 17, 4)
    cache = {"a.jpg": {"objects": [{"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 320, "y2": 180}}]}}
    out = pooler(tokens, original_tokens=tokens, batch={"file_name": ["a.jpg"]}, grounding_cache=cache, image_height=4, image_width=4, patch_size=1)
    assert out["evidence_tokens"].shape[-1] == 4
    assert out["counts"]["object"] > 0
    assert out["evidence_mask"].any()


def test_evidence_pooler_fallback_marks_source():
    pooler = CausalEvidencePooler(dim=4, per_reason_topk_evidence=3)
    tokens = torch.randn(1, 10, 4)
    out = pooler(tokens, original_tokens=tokens, batch={"file_name": ["missing.jpg"]}, grounding_cache={})
    assert out["counts"]["fallback"] == 3
    assert out["meta"][0][0]["source"] == "fallback_attention"
    assert float(out["evidence_quality"].mean()) <= 0.35


def test_box_to_patch_mask():
    mask = box_to_patch_mask({"x1": 0, "y1": 0, "x2": 640, "y2": 360}, 45, 80, 720, 1280)
    assert mask.any()

