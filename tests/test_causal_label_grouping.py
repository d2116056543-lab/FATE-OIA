from __future__ import annotations

import torch

from fate_oia.models.causal_label_grouping import CausalLabelGrouping


def test_causal_label_grouping_identity_init():
    mod = CausalLabelGrouping(dim=8, num_heads=2)
    x = torch.randn(2, 21, 8, requires_grad=True)
    y = mod(x)
    assert y.shape == x.shape
    assert (y - x).abs().mean() < 0.10
    y.sum().backward()
    assert x.grad is not None

