import torch
from fate_oia.models.aie_cert_contribution_head import AIECertContributionHead, stable_signed_decomposition


def test_bias_free_exact_reconstruction():
    m=AIECertContributionHead(dim=16); assert not any('bias' in k for k in m.state_dict())
    x=torch.randn(2,4,4,16); primary=torch.randn(2,4); out=m(x,primary,.7)
    assert out['contribution_reconstruction_error'] < 1e-6
    zero=m(torch.zeros_like(x),primary,.7); assert torch.allclose(zero['raw_contribution'],torch.zeros_like(zero['raw_contribution']))


def test_signed_decomposition_is_finite_when_raw_contributions_cancel():
    raw = torch.tensor([[[1.0, -1.0, 0.5, -0.5]]], requires_grad=True)
    delta = torch.tensor([[0.2]], requires_grad=True)
    bounded = stable_signed_decomposition(raw, delta)

    assert torch.allclose(bounded.sum(-1), delta, atol=1e-7)
    bounded.square().sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.isfinite(delta.grad).all()
