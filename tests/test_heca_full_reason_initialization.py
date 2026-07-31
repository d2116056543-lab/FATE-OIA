import torch

from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder


def test_reason_global_is_exact_calalign_anchor_at_initialization() -> None:
    module = METERPrivateReasonDecoder(dim=8)
    anchor = torch.randn(2, 21)
    out = module(
        reason_logits_calalign=anchor, reason_nodes=torch.randn(2, 21, 8),
        factor_measurement_token=torch.randn(2, 21, 8), factor_reliability=torch.ones(2, 21),
        factor_groundable_mask=torch.ones(21), progress=0.0,
    )
    assert torch.equal(out["reason_logits_global"], anchor)
    assert torch.equal(out["reason_logits_final"], anchor)

