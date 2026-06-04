import torch

from fate_oia.models.p3le_progressive_experts import ProgressiveLayeredExperts


def test_progressive_layered_experts_have_required_levels():
    model = ProgressiveLayeredExperts(dim=32, action_dim=4, reason_dim=21, tail_indices=(5, 6, 9))
    assert hasattr(model, "shared_1")
    assert hasattr(model, "action_1")
    assert hasattr(model, "reason_1")
    assert hasattr(model, "shared_2")
    assert hasattr(model, "action_2")
    assert hasattr(model, "reason_2")
    assert hasattr(model, "tail_2")
    out = model(torch.randn(2, 4, 32), torch.randn(2, 21, 32), torch.randn(2, 32))
    assert tuple(out["action_tokens"].shape) == (2, 4, 32)
    assert tuple(out["reason_tokens"].shape) == (2, 21, 32)
    assert tuple(out["tail_reason_gate"].shape) == (2, 21, 1)
