import torch

from fate_oia.models.tida_action_reader import TIDAActionReader


def test_action_delta_is_bounded():
    model = TIDAActionReader(dim=8, kappa=0.15)
    out = model(torch.randn(2, 4, 8), torch.randn(2, 32, 8) * 100, torch.randn(2, 4, 8) * 100, torch.ones(2, 36), temporal_scale=1.0)
    assert out["action_temporal_delta"].abs().max().item() <= 0.150001
