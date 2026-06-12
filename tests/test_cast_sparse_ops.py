import torch

from fate_oia.models.cast_sparse_ops import entmax15_bisect, sparsemax


def _check_sparse_distribution(fn):
    x = torch.tensor([[2.0, 0.2, -3.0, 0.1]], requires_grad=True)
    p = fn(x, dim=-1)
    assert p.shape == x.shape
    assert torch.all(p >= -1e-6)
    assert torch.allclose(p.sum(-1), torch.ones(1), atol=1e-5)
    assert int((p <= 1e-7).sum().item()) >= 1
    loss = (p * torch.arange(1, 5, dtype=p.dtype)).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_sparsemax_distribution_and_gradients():
    _check_sparse_distribution(sparsemax)


def test_entmax15_distribution_and_gradients():
    _check_sparse_distribution(entmax15_bisect)
