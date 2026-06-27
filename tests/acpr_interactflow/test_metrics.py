from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.psi_metrics import compute_psi_action_metrics, compute_psi_exp29_metrics


def test_action_metrics_include_damo_style_fields():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 3.0]])
    y = torch.tensor([0, 1, 2])
    soft = torch.nn.functional.one_hot(y, 3).float()
    metrics = compute_psi_action_metrics(logits, y, soft)
    assert metrics["Act_mAcc"] > 0.99
    assert "Act_stopF1" in metrics
    assert metrics["Act_stopF1"] == metrics["per_class_f1"][2]
    assert "Act_macro_F1" in metrics
    assert "Stop_F1" in metrics
    assert len(metrics["confusion"]) == 3


def test_exp29_metrics_respect_unknown_mask():
    logits = torch.zeros(2, 29)
    y = torch.zeros(2, 29)
    mask = torch.ones(2, 29)
    mask[0].zero_()
    y[1, 3] = 1
    metrics = compute_psi_exp29_metrics(logits, y, mask)
    assert metrics["all_zero_unknown_count"] == 1
    assert len(metrics["per_label_f1"]) == 29
