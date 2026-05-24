from __future__ import annotations

import torch

from fate_oia.engine.train_fate_oia import (
    compress_tokens,
    reason_to_action_consistency_loss,
    recover_label_attention,
)
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel


def test_compress_tokens_preserves_cls_and_provenance_rows():
    tokens = torch.randn(2, 17, 8)
    reduced, provenance, stats = compress_tokens(tokens, keep_ratio=0.5, num_summary_tokens=2, min_tokens=4)
    assert reduced.shape[0] == 2
    assert reduced.shape[-1] == 8
    assert provenance is not None
    assert provenance.shape[:2] == (2, 17)
    assert torch.allclose(provenance.sum(-1), torch.ones(2, 17), atol=1e-6)
    assert torch.allclose(reduced[:, 0], tokens[:, 0])
    assert stats["enabled"] is True


def test_fate_oia_feature_model_outputs_attention_and_r2a():
    model = FATEOIAFeatureModel(dim=16, action_dim=4, reason_dim=21, use_label_query=True)
    out = model(torch.randn(3, 10, 16))
    assert out["action_logits"].shape == (3, 4)
    assert out["reason_logits"].shape == (3, 21)
    assert out["reason_to_action_logits"].shape == (3, 4)
    assert out["attention"].shape[:3] == (3, 4, 25)


def test_recover_label_attention_from_compressed_tokens():
    tokens = torch.randn(1, 9, 4)
    reduced, provenance, _ = compress_tokens(tokens, keep_ratio=0.5, num_summary_tokens=1, min_tokens=2)
    attention = torch.softmax(torch.randn(1, 2, 25, reduced.shape[1]), dim=-1)
    recovered = recover_label_attention(attention, provenance, original_tokens=tokens.shape[1])
    assert recovered.shape == (1, 25, 9)


def test_reason_to_action_consistency_loss_is_scalar():
    loss = reason_to_action_consistency_loss(torch.randn(5, 4), torch.randn(5, 4))
    assert loss.ndim == 0
    assert torch.isfinite(loss)