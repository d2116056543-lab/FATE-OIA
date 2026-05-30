from __future__ import annotations

import torch

from fate_oia.engine.calibrate_cafe_oia import apply_calibration, fit_classwise_bias_temperature, threshold_sweep_per_label


def test_calibration_params_non_placeholder_and_changes_logits() -> None:
    logits = torch.tensor([[-2.0, 0.1], [0.2, -1.0], [1.0, -0.2], [-0.5, 2.0]])
    labels = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = fit_classwise_bias_temperature(logits, labels)
    assert len(params["bias"]) == 2
    assert len(params["temperature"]) == 2
    assert "placeholder" not in str(params).lower()
    calibrated = apply_calibration(logits, params)
    assert float((calibrated - logits).abs().sum()) > 0


def test_threshold_sweep_per_label() -> None:
    logits = torch.tensor([[-2.0, 0.1], [0.2, -1.0], [1.0, -0.2], [-0.5, 2.0]])
    labels = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out = threshold_sweep_per_label(logits, labels)
    assert len(out["thresholds"]) == 2
    assert out["macro_f1"] >= 0
