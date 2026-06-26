from __future__ import annotations

import torch

from fate_oia.utils.acpr_vista_gradient_coordinator import common_descent_direction


def test_common_descent_handles_conflict():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([-0.5, 1.0])
    g, stats = common_descent_direction(a, b)
    assert torch.dot(g, a) >= -1e-8
    assert torch.dot(g, b) >= -1e-8
    assert stats["adapter_gradient_conflict"] is True

