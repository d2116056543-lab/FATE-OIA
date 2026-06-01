from __future__ import annotations

import torch

from fate_oia.models.sure_relation_attention import SURESparseRelationAttention


def test_sparse_relation_attention_selects_subset() -> None:
    attn = SURESparseRelationAttention(dim=32, label_count=5, max_edges_per_label=2, max_edges_total=6)
    label_tokens = torch.randn(2, 5, 32)
    rel_tokens = torch.randn(2, 8, 32)
    out = attn(label_tokens, rel_tokens)
    assert out["label_context"].shape == label_tokens.shape
    assert out["selected_relation_indices"].shape == (2, 6)
    assert out["stats"]["selected_edges"] < out["stats"]["candidate_edges"]
