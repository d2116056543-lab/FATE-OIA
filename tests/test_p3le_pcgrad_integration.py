from pathlib import Path

import torch

from fate_oia.losses.pcgrad_lite import apply_pcgrad_lite


def test_train_script_explicitly_applies_pcgrad_lite():
    src = Path("fate_oia/engine/train_p3le_pair_oia.py").read_text(encoding="utf-8")
    assert "apply_pcgrad_lite(" in src
    assert "shared_params = list(model.shared_parameters_for_budget())" in src


def test_apply_pcgrad_lite_assigns_shared_parameter_gradients():
    layer = torch.nn.Linear(4, 1)
    x = torch.randn(5, 4)
    y1 = layer(x).mean()
    y2 = -layer(x).mean()
    stats = apply_pcgrad_lite([y1, y2], layer.parameters(), retain_graph=True)
    assert "projection_applied_count" in stats
    assert all(p.grad is not None for p in layer.parameters())

