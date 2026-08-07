import torch
from fate_oia.models.aie_cert_atom_transport import AIECertAtomTransport


def test_map_token_share_transport_matrix():
    module = AIECertAtomTransport(dim=16, heads=4)
    token, amap = torch.randn(2,4,4,16), torch.softmax(torch.randn(2,4,4,20), -1)
    out = module(token, amap)
    matrix = out['atom_transport_matrix']
    assert torch.allclose(matrix.diagonal(dim1=-2, dim2=-1), torch.zeros(2,4,4), atol=1e-6)
    assert torch.allclose(matrix.sum(-1), torch.ones(2,4,4), atol=1e-5)
    assert (out['atom_map'] - amap).abs().sum() > 0 and (out['atom_token'] - token).abs().sum() > 0
