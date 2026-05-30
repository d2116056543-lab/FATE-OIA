from __future__ import annotations

import torch

from fate_oia.models.causal_evidence_pooler import CausalEvidencePooler


def test_object_box_evidence_nonzero() -> None:
    pooler = CausalEvidencePooler(dim=8, per_reason_topk_evidence=2)
    tokens = torch.randn(1, 1 + 45 * 80, 8)
    batch = {"file_name": ["sample.jpg"]}
    cache = {"sample.jpg": {"objects": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 200, "y2": 200}}]}}
    out = pooler(tokens, batch=batch, grounding_cache=cache, reason_rules={5: {"car"}}, image_height=360, image_width=640, patch_size=8)
    assert out["counts"]["object"] > 0
    assert int(out["real_evidence_mask"].sum()) > 0
    assert bool(out["reason_evidence_mask"][0, 5].any())


def test_lane_and_drivable_poly_synthetic() -> None:
    pooler = CausalEvidencePooler(dim=8, per_reason_topk_evidence=2)
    tokens = torch.randn(1, 1 + 45 * 80, 8)
    poly = [{"vertices": [[0, 0], [100, 0], [100, 100], [0, 100]]}]
    cache = {"sample.jpg": {"objects": [{"category": "lane/crosswalk", "poly2d": poly}, {"category": "area/drivable", "poly2d": poly}]}}
    out = pooler(tokens, batch={"file_name": ["sample.jpg"]}, grounding_cache=cache, reason_rules={11: {"lane/crosswalk"}, 2: {"area/drivable"}})
    assert out["counts"]["lane"] > 0
    assert out["counts"]["drivable"] > 0
    assert bool(out["reason_evidence_mask"][0, 11].any())
    assert bool(out["reason_evidence_mask"][0, 2].any())


def test_fallback_flagged_and_low_quality() -> None:
    pooler = CausalEvidencePooler(dim=8, per_reason_topk_evidence=2, fallback_quality_multiplier=0.20)
    tokens = torch.randn(1, 1 + 45 * 80, 8)
    out = pooler(tokens, batch={"file_name": ["missing.jpg"]}, grounding_cache={})
    assert out["counts"]["fallback"] == 2
    assert not bool(out["real_evidence_mask"].any())
    assert torch.all(out["evidence_source_type"] == 3)
