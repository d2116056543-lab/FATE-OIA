import torch
from fate_oia.models.aie_cert_sparse import entmax15


def test_entmax_contract_and_gradient():
    x = torch.tensor([[3.0, 1.0, -4.0], [0.0, 0.0, 0.0]], requires_grad=True)
    p = entmax15(x)
    assert torch.allclose(p.sum(-1), torch.ones(2))
    assert (p >= 0).all() and (p == 0).any()
    assert torch.allclose(p[1], torch.full((3,), 1 / 3))
    p.square().sum().backward(); assert torch.isfinite(x.grad).all()
