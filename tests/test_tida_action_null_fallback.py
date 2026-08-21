import torch

from fate_oia.models.tida_action_reader import TIDAActionReader


def test_zero_reliability_forces_exact_zero_delta():
    model = TIDAActionReader(dim=8)
    out = model(torch.randn(2, 4, 8), torch.randn(2, 32, 8), torch.randn(2, 4, 8), torch.zeros(2, 36), temporal_scale=1.0)
    assert torch.equal(out["action_temporal_delta"], torch.zeros_like(out["action_temporal_delta"]))
