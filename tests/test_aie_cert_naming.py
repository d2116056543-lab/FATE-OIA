import torch
from fate_oia.models.aie_cert_naming import AIECertNaming


def test_naming_has_no_keys_and_detaches_shared_inputs():
    m=AIECertNaming(dim=16); assert 'predicate_keys' not in dict(m.named_parameters())
    token=torch.randn(1,4,4,16,requires_grad=True); keys=torch.randn(32,64,requires_grad=True)
    out=m(token,torch.softmax(torch.randn(1,4,4,20),-1),keys,torch.softmax(torch.randn(1,32,20),-1),torch.rand(1,32))
    out['name_quality'].sum().backward(); assert token.grad is None and keys.grad is None
    assert m.projection.weight.grad is not None
