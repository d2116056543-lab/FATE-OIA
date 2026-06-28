from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.model import bounded_exp29_calalign_delta

from fate_oia.acpr_interactflow.calibrated_exp29 import (
    exp29_calibration_quality,
    fit_exp29_theta_from_train_logits,
    should_accept_exp29_theta,
)

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



def test_train_only_theta_fit_creates_fixed_threshold_positives_without_test_labels() -> None:
    logits = torch.tensor([[-2.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([[0, 0], [0, 1], [1, 0], [0, 0]], dtype=torch.float32)
    mask = torch.ones_like(targets)

    theta, rates = fit_exp29_theta_from_train_logits(logits, targets, mask, pi_min=0.25, pi_max=0.50)
    theta_with_margin, _ = fit_exp29_theta_from_train_logits(
        logits,
        targets,
        mask,
        pi_min=0.25,
        pi_max=0.50,
        deploy_logit_margin=0.25,
    )
    calibrated = logits - theta.view(1, -1)
    calibrated_with_margin = logits - theta_with_margin.view(1, -1)
    pred_rate = (torch.sigmoid(calibrated) >= 0.5).float().mean(0)
    pred_rate_with_margin = (torch.sigmoid(calibrated_with_margin) >= 0.5).float().mean(0)

    assert torch.isfinite(theta).all()
    assert torch.all(rates >= 0.25)
    assert torch.all(pred_rate > 0.0)
    assert torch.all(pred_rate <= 0.50)
    assert torch.all(theta_with_margin < theta)
    assert torch.all(pred_rate_with_margin >= pred_rate)



def test_train_calibration_rejects_candidate_that_collapses_fixed_threshold_predictions() -> None:
    logits = torch.tensor([[1.0, -1.0], [0.8, -0.8], [-0.5, 1.2], [-0.2, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float32)
    mask = torch.ones_like(targets)

    current_theta = torch.zeros(2)
    collapsed_theta = torch.tensor([10.0, 10.0])
    current = exp29_calibration_quality(logits, targets, mask, current_theta)
    collapsed = exp29_calibration_quality(logits, targets, mask, collapsed_theta)

    assert current["mF1"] > collapsed["mF1"]
    assert collapsed["pred_positive_rate"] == 0.0
    assert not should_accept_exp29_theta(collapsed, current, min_pred_positive_rate=0.02)



def test_theta_fit_controls_global_unknown_prediction_rate_not_only_valid_mask() -> None:
    logits = torch.tensor([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=torch.float32)

    theta, rates = fit_exp29_theta_from_train_logits(logits, targets, mask, pi_min=0.10, pi_max=0.20)
    global_pred_rate = (torch.sigmoid(logits - theta.view(1, -1)) >= 0.5).float().mean()

    assert torch.all(rates <= 0.20)
    assert global_pred_rate <= 0.35



def test_exp29_second_stage_calalign_delta_cannot_override_train_calib_threshold() -> None:
    raw = torch.zeros(2, 3)
    learned_cal = torch.full((2, 3), 2.0)
    delta = bounded_exp29_calalign_delta(learned_cal, raw, max_delta=0.05)

    assert delta.abs().max() <= 0.050001
