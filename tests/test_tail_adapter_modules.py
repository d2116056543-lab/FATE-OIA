from __future__ import annotations

import json
from pathlib import Path

import torch

from fate_oia.losses.tail_ranking_loss import tail_margin_ranking_loss
from fate_oia.models.frozen_run_c_predictor import load_run_c_cache
from fate_oia.models.tail_calibration import PerLabelBiasCalibrator, thresholds_to_bias
from fate_oia.models.tail_residual_adapter import TailResidualAdapter
from fate_oia.utils.config_fingerprint import config_fingerprint, diff_configs


def test_config_fingerprint_is_stable_and_diff_is_nested() -> None:
    cfg_a = {"model": {"width": 640, "height": 360}, "loss": {"lr": 1e-4}}
    cfg_b = {"loss": {"lr": 1e-4}, "model": {"height": 360, "width": 640}}
    assert config_fingerprint(cfg_a)["sha256"] == config_fingerprint(cfg_b)["sha256"]

    cfg_c = {"model": {"width": 512, "height": 360}, "loss": {"lr": 1e-4}}
    diff = diff_configs(cfg_a, cfg_c)
    assert diff["changed"]["model.width"] == {"left": 640, "right": 512}


def test_run_c_cache_loads_logits_labels_and_file_names(tmp_path: Path) -> None:
    torch.save(torch.randn(3, 4), tmp_path / "logits_action_fused_best_test.pt")
    torch.save(torch.randn(3, 4), tmp_path / "logits_action_visual_best_test.pt")
    torch.save(torch.randn(3, 4), tmp_path / "logits_action_reason_best_test.pt")
    torch.save(torch.randn(3, 21), tmp_path / "logits_reason_best_test.pt")
    torch.save(torch.zeros(3, 4), tmp_path / "labels_action_best_test.pt")
    torch.save(torch.zeros(3, 21), tmp_path / "labels_reason_best_test.pt")
    (tmp_path / "file_names_best_test.json").write_text(json.dumps(["a.jpg", "b.jpg", "c.jpg"]), encoding="utf-8")

    cache = load_run_c_cache(tmp_path, suffix="best_test")
    assert cache.action_fused_logits.shape == (3, 4)
    assert cache.reason_logits.shape == (3, 21)
    assert cache.labels.shape == (3, 25)
    assert cache.file_names == ["a.jpg", "b.jpg", "c.jpg"]


def test_per_label_bias_calibrator_zero_init_is_identity_and_threshold_bias_matches_decision() -> None:
    logits = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]])
    calibrator = PerLabelBiasCalibrator(num_labels=3)
    assert torch.allclose(calibrator(logits), logits)

    thresholds = torch.tensor([0.2, 0.5, 0.8])
    bias = thresholds_to_bias(thresholds)
    calibrator = PerLabelBiasCalibrator(num_labels=3, init_bias=bias)
    calibrated_pred = torch.sigmoid(calibrator(logits)) >= 0.5
    threshold_pred = torch.sigmoid(logits) >= thresholds.view(1, -1)
    assert torch.equal(calibrated_pred, threshold_pred)


def test_tail_residual_adapter_starts_as_identity_and_only_changes_tail_labels() -> None:
    base_action = torch.randn(5, 4)
    base_reason = torch.randn(5, 21)
    tail_indices = [12, 9, 5, 14, 6, 11, 10, 13]
    adapter = TailResidualAdapter(action_dim=4, reason_dim=21, tail_indices=tail_indices, hidden_dim=16)

    out = adapter(base_action, base_reason)
    assert torch.allclose(out["reason_logits"], base_reason)
    assert torch.allclose(out["delta_reason_logits"], torch.zeros_like(base_reason))

    with torch.no_grad():
        adapter.delta_scale.fill_(0.5)
    out = adapter(base_action, base_reason)
    non_tail = sorted(set(range(21)) - set(tail_indices))
    assert torch.allclose(out["delta_reason_logits"][:, non_tail], torch.zeros(5, len(non_tail)))


def test_tail_margin_ranking_loss_is_finite_and_backpropagates() -> None:
    logits = torch.tensor([[0.2, -0.1, 0.9], [-0.4, 0.7, 0.1]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss = tail_margin_ranking_loss(logits, labels, tail_indices=[0, 1, 2], margin=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0
