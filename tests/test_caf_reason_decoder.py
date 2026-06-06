import torch
from fate_oia.models.caf_reason_decoder import MaskedReasonFromFactorTransformer

def test_reason_decoder_outputs_attention_and_bounded_delta():
    dec = MaskedReasonFromFactorTransformer(dim=32, action_dim=4, reason_dim=21, reason_cap=0.25)
    selected = torch.randn(2,4,3,32)
    actor = torch.randn(2,4,3,32)
    base = torch.zeros(2,21)
    out = dec(selected, actor, base)
    assert out['final_reason_logits'].shape == (2,21)
    assert out['reason_to_factor_attention'].shape[:2] == (2,21)
    assert (out['final_reason_logits'] - base).abs().max() <= 0.2501
