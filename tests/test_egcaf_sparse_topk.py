import torch
from fate_oia.models.egcaf_sparse_topk import sparsemax, entmax15, relaxed_topk


def test_sparsemax_zero_and_sum_and_grad():
    s = torch.tensor([[4.0, 1.0, -2.0]], requires_grad=True)
    p = sparsemax(s, -1)
    assert torch.allclose(p.sum(-1), torch.ones(1))
    assert (p == 0).any()
    p.sum().backward()
    assert s.grad is not None


def test_entmax_and_relaxed_topk_shape():
    s = torch.randn(2, 4, 9, requires_grad=True)
    w = entmax15(s, -1)
    idx, vals = relaxed_topk(w, 3)
    assert idx.shape == (2, 4, 3)
    assert vals.shape == (2, 4, 3)
