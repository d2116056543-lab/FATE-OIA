from __future__ import annotations

import torch

from fate_oia.models.psr_oia_router import StaticLabelRouter


def test_static_router_is_per_label_not_plain_average():
    torch.manual_seed(1)
    action_a, action_e = torch.randn(5, 4), torch.randn(5, 4)
    reason_a, reason_e = torch.randn(5, 21), torch.randn(5, 21)
    source = ["E" if i % 2 else "A" for i in range(21)]
    out = StaticLabelRouter(source)(action_a, reason_a, action_e, reason_e)
    assert out.action_logits.shape == (5, 4)
    assert out.reason_logits.shape == (5, 21)
    assert not torch.allclose(out.reason_logits, 0.5 * reason_a + 0.5 * reason_e)
    assert torch.allclose(out.reason_logits[:, 1], reason_e[:, 1])
    assert torch.allclose(out.reason_logits[:, 0], reason_a[:, 0])
