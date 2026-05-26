import json
from pathlib import Path

import torch

from fate_oia.engine.offline_fusion_alpha_sweep import run_alpha_sweep


def test_alpha_sweep_prefers_reason_when_visual_is_bad(tmp_path: Path):
    reason = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    visual = -reason
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reason_logits = torch.zeros(2, 3)
    reason_labels = torch.zeros(2, 3)
    result = run_alpha_sweep(
        visual_action_logits=visual,
        reason_action_logits=reason,
        labels_action=labels,
        reason_logits=reason_logits,
        labels_reason=reason_labels,
        output_dir=tmp_path,
        action_dim=2,
        alphas=[0.0, 0.5, 1.0],
    )
    assert result["best_alpha"] == 0.0
    assert result["fusion_fix_recommended"] is False
    assert (tmp_path / "alpha_sweep_test.json").exists()
    json.loads((tmp_path / "alpha_sweep_test.json").read_text())

