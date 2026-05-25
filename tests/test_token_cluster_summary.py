from __future__ import annotations

import torch

from fate_oia.models.token_provenance import keep_merge_tokens


def test_multiple_summary_tokens_are_not_duplicate_means():
    torch.manual_seed(0)
    tokens = torch.randn(1, 12, 6)
    scores = torch.arange(12, dtype=torch.float32).view(1, 12)
    reduced, provenance, stats = keep_merge_tokens(tokens, scores=scores, keep_ratio=0.5, num_summary_tokens=3, min_tokens=1)
    summary = reduced[:, -3:, :]
    assert stats["summary_tokens"] == 3
    assert provenance.shape == (1, 12, reduced.shape[1])
    assert torch.allclose(provenance.sum(-1), torch.ones(1, 12), atol=1e-6)
    assert not torch.allclose(summary[:, 0], summary[:, 1])


def test_single_summary_token_remains_mean_like():
    torch.manual_seed(1)
    tokens = torch.randn(1, 8, 4)
    scores = torch.arange(8, dtype=torch.float32).view(1, 8)
    reduced, provenance, stats = keep_merge_tokens(tokens, scores=scores, keep_ratio=0.5, num_summary_tokens=1, min_tokens=1)
    assert stats["summary_tokens"] == 1
    assert torch.allclose(provenance.sum(-1), torch.ones(1, 8), atol=1e-6)
