from __future__ import annotations

import torch
from fate_oia.models.eagle_pu_sparse_ops import entmax15_bisect, sparsemax

def _check_distribution(p: torch.Tensor) -> None:
    assert torch.all(torch.isfinite(p))
    assert torch.all(p >= -1e-6)
    assert torch.allclose(p.sum(dim=-1), torch.ones(p.shape[:-1]), atol=1e-5)

def test_sparsemax_has_zeros_and_finite_gradients():
    x = torch.tensor([[3.0, 1.0, -2.0, 0.5]], requires_grad=True)
    p = sparsemax(x, dim=-1)
    _check_distribution(p)
    assert (p < 1e-7).any()
    p[..., 0].sum().backward()
    assert torch.all(torch.isfinite(x.grad))

def test_entmax15_has_zeros_and_finite_gradients():
    x = torch.tensor([[3.0, 1.0, -2.0, 0.5]], requires_grad=True)
    p = entmax15_bisect(x, dim=-1, n_iter=40)
    _check_distribution(p)
    assert (p < 1e-7).any()
    p[..., 0].sum().backward()
    assert torch.all(torch.isfinite(x.grad))
