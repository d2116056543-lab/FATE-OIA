from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.models.cafe_oia_model import CAFEOIAModel


def test_counterfactual_recomputes_forward(monkeypatch) -> None:
    model = CAFEOIAModel(dim=16, action_dim=4, reason_dim=21)
    calls = {"n": 0}
    original = model.forward_from_base_and_evidence

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_from_base_and_evidence", wrapped)
    tokens = torch.randn(1, 1 + 45 * 80, 16)
    labels = torch.zeros(1, 21)
    labels[0, 5] = 1.0
    cache = {"a.jpg": {"objects": [{"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 200, "y2": 200}}]}}
    out = model(tokens, batch={"file_name": ["a.jpg"]}, grounding_cache=cache, return_cf=True, cf_targets=labels, reason_rules={5: {"car"}})
    assert calls["n"] >= 5
    assert not out["cf"]["cf_is_proxy"]
    assert int(out["cf"]["cf_valid_mask"].sum()) > 0


def test_no_proxy_string() -> None:
    text = Path("fate_oia/models/cafe_oia_model.py").read_text(encoding="utf-8")
    assert "target_deleted_reason = reason_logits - torch.relu" not in text


def test_fallback_only_skips_cf() -> None:
    model = CAFEOIAModel(dim=16, action_dim=4, reason_dim=21, allow_fallback_counterfactual=False)
    tokens = torch.randn(1, 1 + 45 * 80, 16)
    labels = torch.zeros(1, 21)
    labels[0, 5] = 1.0
    out = model(tokens, batch={"file_name": ["missing.jpg"]}, grounding_cache={}, return_cf=True, cf_targets=labels, reason_rules={5: {"car"}})
    assert int(out["cf"]["cf_real_evidence_mask"].sum()) == 0
