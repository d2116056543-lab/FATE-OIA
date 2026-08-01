import torch

from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder


def test_reason_correction_off_is_global() -> None:
    decoder = METERPrivateReasonDecoder(dim=32)
    out = decoder(
        reason_logits_calalign=torch.randn(2, 21),
        reason_nodes=torch.randn(2, 21, 32),
        factor_measurement_token=torch.randn(2, 21, 32),
        factor_reliability=torch.zeros(2, 21),
        factor_groundable_mask=torch.ones(21),
        progress=1,
    )
    torch.testing.assert_close(out["reason_logits_final"], out["reason_logits_global"])
