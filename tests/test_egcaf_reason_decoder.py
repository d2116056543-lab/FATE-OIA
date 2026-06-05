import torch
from fate_oia.models.egcaf_reason_decoder import ReasonFromFactorDecoder


def test_reason_decoder_consumes_selected_factors():
    dec = ReasonFromFactorDecoder(hidden_dim=32)
    selected = torch.randn(2,4,3,32)
    scene = torch.randn(2,6,32)
    out = dec(selected, scene)
    assert out["reason_logits"].shape == (2,21)
    out2 = dec(selected + 1.0, scene)
    assert not torch.allclose(out["reason_logits"], out2["reason_logits"])
