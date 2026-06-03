from __future__ import annotations

import torch

from fate_oia.engine.train_fate_oia import empty_eval_stats, parse_eval_splits
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.label_correlation import LabelCorrelationBlock


def test_label_correlation_block_shape_attention_and_gradient():
    torch.manual_seed(0)
    block = LabelCorrelationBlock(dim=16, num_layers=2, num_heads=4, dropout=0.0)
    tokens = torch.randn(3, 25, 16, requires_grad=True)
    out = block(tokens)
    assert out["label_tokens"].shape == (3, 25, 16)
    assert out["attention"].shape == (3, 2, 4, 25, 25)
    loss = out["label_tokens"].pow(2).mean()
    loss.backward()
    assert tokens.grad is not None
    assert float(tokens.grad.abs().sum()) > 0


def test_fate_oia_label_correlation_changes_logits_and_records_state():
    torch.manual_seed(1)
    tokens = torch.randn(2, 12, 16)
    base = FATEOIAFeatureModel(dim=16, action_dim=4, reason_dim=21, use_label_query=True)
    corr = FATEOIAFeatureModel(
        dim=16,
        action_dim=4,
        reason_dim=21,
        use_label_query=True,
        use_label_correlation=True,
        label_correlation_layers=1,
        label_correlation_heads=4,
        label_correlation_dropout=0.0,
        fusion_mode="visual",
    )
    out_base = base(tokens)
    out_corr = corr(tokens)
    assert out_corr["action_fused_logits"].shape == (2, 4)
    assert out_corr["reason_logits"].shape == (2, 21)
    assert out_corr["label_correlation_attention"].shape == (2, 1, 4, 25, 25)
    assert bool(out_corr["label_correlation_enabled"].item()) is True
    assert torch.allclose(out_corr["fusion_gate"], torch.ones_like(out_corr["fusion_gate"]))
    assert not torch.allclose(out_base["reason_logits"], out_corr["reason_logits"])


def test_eval_splits_parser_and_empty_stats_are_test_only_safe():
    class Args:
        action_dim = 4
        reason_dim = 21

    assert parse_eval_splits("test") == {"test"}
    stats = empty_eval_stats(Args())
    assert stats["logits"].shape == (0, 25)
    assert stats["fused_logits"].shape == (0, 4)
