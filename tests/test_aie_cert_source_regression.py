import torch
from fate_oia.models.aie_calalign_foundation import AIECalAlignFoundation
from fate_oia.models.aie_cert_calalign_foundation import AIECertCalAlignFoundation


def test_cert_foundation_is_value_equivalent_to_source():
    torch.manual_seed(7); source=AIECalAlignFoundation(use_mock_dino=True,mock_dim=384)
    cert=AIECertCalAlignFoundation(use_mock_dino=True,mock_dim=384); cert.load_state_dict(source.state_dict())
    image=torch.randn(1,3,360,640); a=source(image); b=cert(image)
    assert torch.allclose(a['action_logits_primary'],b['action_logits_primary'],atol=1e-6)
    assert torch.allclose(a['reason_logits_primary'],b['reason_logits_primary'],atol=1e-6)
