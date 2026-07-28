import torch

from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder


def test_global_reason_view_is_directly_computed() -> None:
    decoder = METERPrivateReasonDecoder(dim=16, reason_dim=21, action_dim=4)
    output = decoder(
        patch_tokens_by_layer=torch.randn(2, 3, 12, 16),
        reason_logits_calalign=torch.randn(2, 21),
        action_logits_final=torch.randn(2, 4),
        action_nodes=torch.randn(2, 4, 16),
        factor_to_reason_tokens=torch.randn(2, 21, 16),
        factor_support_map=torch.softmax(torch.randn(2, 21, 12), -1),
        factor_counter_map=torch.softmax(torch.randn(2, 21, 12), -1),
        factor_reliability=torch.rand(2, 21),
        factor_support_null=torch.rand(2, 21),
        progress=1.0,
    )
    assert output["reason_logits_global"].shape == (2, 21)
    assert torch.isfinite(output["reason_logits_global"]).all()
