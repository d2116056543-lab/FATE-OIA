from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.models.mosaic_group_threshold import MOSAICGroupThresholdHead


def test_calibration_masks_labels_without_positive_support_from_data_terms() -> None:
    head = MOSAICGroupThresholdHead()
    action_logits = torch.randn(8, 4)
    reason_logits = torch.randn(8, 21)
    action_targets = torch.ones(8, 4)
    reason_targets = torch.ones(8, 21)
    reason_targets[:, 7] = 0.0
    valid = torch.cat((action_targets, reason_targets), dim=1).sum(0).gt(0)

    first = head.calibration_objective(
        action_logits, reason_logits, action_targets, reason_targets, valid_label_mask=valid
    )
    changed_reason_logits = reason_logits.clone()
    changed_reason_logits[:, 7] = 100.0
    second = head.calibration_objective(
        action_logits, changed_reason_logits, action_targets, reason_targets, valid_label_mask=valid
    )

    assert valid.sum().item() == 24
    for key in ("loss_calibration_soft_f1", "loss_calibration_bce", "loss_calibration_rate", "loss_calibration_total"):
        assert torch.allclose(first[key], second[key], atol=1e-7, rtol=0.0), key


def test_trainer_passes_support_mask_and_records_unsupported_labels() -> None:
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert "valid_label_mask=positive_support" in source
    assert '"unsupported_label_ids"' in source
    assert "train_calib lacks positive support" not in source
