from __future__ import annotations

import torch

from fate_oia.models.acpr_visual_token_adapter import ACPRPredicateAnchoredVisualAdapter


def test_vista_adapter_zero_gate_preserves_tokens_and_shapes():
    adapter = ACPRPredicateAnchoredVisualAdapter(dim=384, rank=48, num_layers=3, num_predicates=32)
    x = torch.randn(2, 3, 3600, 384)
    probs = torch.rand(2, 32)
    attn = torch.softmax(torch.randn(2, 32, 3600), dim=-1)
    y, stats = adapter(x, probs, attn, epoch=0)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert stats["vista_gate_map"].shape == (2, 3600)
    assert float(stats["vista_gate_mean"]) >= 0.20


def test_vista_adapter_gate_gradient_is_not_dead():
    adapter = ACPRPredicateAnchoredVisualAdapter(dim=384, rank=48, num_layers=3, num_predicates=32)
    x = torch.randn(1, 3, 3600, 384)
    probs = torch.rand(1, 32)
    attn = torch.softmax(torch.randn(1, 32, 3600), dim=-1)
    y, _ = adapter(x, probs, attn, epoch=0)
    loss = y.square().mean()
    loss.backward()
    assert adapter.gate_raw.grad is not None
    assert adapter.gate_raw.grad.abs().sum() > 0

