import inspect

import torch
import pytest

from fate_oia.engine.train_acpr_mosaic_trust_icdor import (
    assert_icdor_gradient_firewall,
    clip_icdor_owner_gradients,
    train_icdor_epoch,
)


def test_per_owner_gradient_clip_is_applied():
    p1 = torch.nn.Parameter(torch.ones(2))
    p2 = torch.nn.Parameter(torch.ones(2))
    p1.grad = torch.tensor([10.0, 0.0])
    p2.grad = torch.tensor([0.1, 0.0])
    norms = clip_icdor_owner_gradients({"action_visual": [p1], "factor": [p2]}, {"action_visual": 1.0, "factor": 0.5})
    assert float(p1.grad.norm()) <= 1.0 + 1e-6
    assert norms["action_visual"] > norms["factor"]


def test_training_epoch_does_not_apply_a_global_clip_after_owner_clipping() -> None:
    source = inspect.getsource(train_icdor_epoch)
    assert "clip_grad_norm_" not in source


def test_gradient_firewall_rejects_forbidden_reason_to_action_edge():
    with pytest.raises(RuntimeError, match="gradient firewall"):
        assert_icdor_gradient_firewall([{
            "loss": "loss_reason_total", "owner_group": "action_adapter", "grad_norm": 1e-3,
        }])
