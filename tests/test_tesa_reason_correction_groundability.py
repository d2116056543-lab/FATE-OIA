import torch

from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder


def test_reason_correction_is_bounded_by_groundability_and_reliability() -> None:
    decoder = METERPrivateReasonDecoder(dim=32)
    out = decoder(
        patch_tokens_by_layer=torch.randn(2, 3, 20, 32),
        reason_logits_calalign=torch.randn(2, 21),
        factor_typed_token=torch.randn(2, 21, 32),
        factor_reliability=torch.ones(2, 21),
        factor_groundable_mask=torch.tensor([1.0] * 14 + [0.0] + [1.0] * 5 + [0.0]),
        progress=1,
    )
    assert out["reason_evidence_delta"][..., [14, 20]].eq(0).all()
    assert (out["reason_evidence_delta"].abs() <= out["reason_correction_kappa"] + 1e-6).all()
