import torch
from fate_oia.models.aie_cert_contribution_head import AIECertContributionHead


def test_bias_free_exact_reconstruction():
    m=AIECertContributionHead(dim=16); assert not any('bias' in k for k in m.state_dict())
    x=torch.randn(2,4,4,16); primary=torch.randn(2,4); out=m(x,primary,.7)
    assert out['contribution_reconstruction_error'] < 1e-6
    zero=m(torch.zeros_like(x),primary,.7); assert torch.allclose(zero['raw_contribution'],torch.zeros_like(zero['raw_contribution']))
