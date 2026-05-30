from __future__ import annotations

import torch

from fate_oia.models.cafe_oia_model import CAFEOIAModel


def test_cafe_initial_near_base():
    model = CAFEOIAModel(dim=16, action_dim=4, reason_dim=21)
    tokens = torch.randn(2, 32, 16)
    out = model(tokens, batch={"file_name": ["a", "b"]}, grounding_cache={})
    assert out["action_logits"].shape == (2, 4)
    assert out["reason_logits"].shape == (2, 21)
    assert (out["reason_logits"] - out["base_reason_logits"]).abs().mean() < 0.25
    assert (out["action_logits"] - out["action_reason_logits"]).abs().mean() < 0.08


def test_visual_residual_uses_evidence():
    model = CAFEOIAModel(dim=16, action_dim=4, reason_dim=21)
    tokens = torch.randn(1, 32, 16)
    out1 = model(tokens, batch={"file_name": ["a"]}, grounding_cache={})
    out2 = model(tokens + 0.5, batch={"file_name": ["a"]}, grounding_cache={})
    assert not torch.allclose(out1["reason_delta_logits"], out2["reason_delta_logits"])


def test_action_residual_bounded():
    model = CAFEOIAModel(dim=16, action_dim=4, reason_dim=21)
    tokens = torch.randn(2, 32, 16)
    out = model(tokens, batch={"file_name": ["a", "b"]}, grounding_cache={})
    assert out["action_delta_logits"].abs().max() <= 0.061

