import torch

from fate_oia.models.tida_temporal_encoder import TIDATemporalEncoder


def test_continuous_timestamp_changes_output():
    model = TIDATemporalEncoder(dim=8, num_layers=1, num_heads=2, dropout=0.0).eval()
    x = torch.randn(1, 14, 1, 8)
    valid = torch.ones(1, 15, dtype=torch.bool)
    a = model(x, torch.linspace(-5, 0, 15).unsqueeze(0), valid)["history_summary"]
    b = model(x, torch.linspace(-4, 0, 15).unsqueeze(0), valid)["history_summary"]
    assert not torch.allclose(a, b)
