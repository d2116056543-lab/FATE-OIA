import torch

from fate_oia.models.p3le_sparse_attention import SparseRegionAttention


def test_sparse_region_attention_topk_shape():
    layer = SparseRegionAttention(dim=32, topk=4)
    out = layer(torch.randn(2, 21, 32), torch.randn(2, 13, 32))
    assert tuple(out["pooled"].shape) == (2, 21, 32)
    assert tuple(out["indices"].shape) == (2, 21, 4)
    assert tuple(out["weights"].shape) == (2, 21, 4)
    assert torch.allclose(out["weights"].sum(dim=-1), torch.ones(2, 21), atol=1e-5)
