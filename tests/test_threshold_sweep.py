from pathlib import Path

import torch

from fate_oia.engine.offline_threshold_sweep import run_threshold_sweep


def test_threshold_sweep_writes_per_label_thresholds(tmp_path: Path):
    logits = torch.tensor([[-2.0, 0.2], [-1.0, 0.4], [3.0, 0.6], [4.0, 0.8]])
    labels = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    result = run_threshold_sweep(logits, labels, output_dir=tmp_path, prefix="Exp")
    assert "fixed" in result
    assert "global" in result
    assert "per_label" in result
    assert len(result["per_label"]["thresholds"]) == 2
    assert (tmp_path / "per_label_thresholds_test.json").exists()

