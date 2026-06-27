from __future__ import annotations

import torch

from fate_oia.losses.acpr_interactflow_losses import (
    exp29_cardinality_loss,
    exp29_masked_asl_loss,
    exp29_positive_rate_loss,
    exp29_soft_f1_loss,
)


def test_exp29_calibrated_logits_drive_fixed_threshold_losses() -> None:
    raw = torch.zeros(3, 4)
    calibrated = raw.clone()
    calibrated[:, 0] = 2.0
    calibrated[:, 1] = -2.0
    targets = torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.float32)

    raw_asl = exp29_masked_asl_loss(raw, targets, mask)
    cal_asl = exp29_masked_asl_loss(calibrated, targets, mask)
    rate = exp29_positive_rate_loss(calibrated, targets, mask)
    card = exp29_cardinality_loss(calibrated, targets, mask)
    soft_f1 = exp29_soft_f1_loss(calibrated, targets, mask)

    assert torch.isfinite(raw_asl)
    assert torch.isfinite(cal_asl)
    assert torch.isfinite(rate)
    assert torch.isfinite(card)
    assert torch.isfinite(soft_f1)
    assert not torch.allclose(raw_asl, cal_asl)
    assert (calibrated.sigmoid() >= 0.5).any()
