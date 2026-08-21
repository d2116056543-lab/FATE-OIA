import torch

from fate_oia.models.tida_temporal_encoder import TIDATemporalEncoder


def test_future_change_does_not_change_earlier_causal_state():
    torch.manual_seed(3)
    model = TIDATemporalEncoder(dim=16, num_layers=2, num_heads=4, dropout=0.0).eval()
    tokens = torch.randn(1, 14, 2, 16)
    timestamps = torch.linspace(-5, 0, 15).unsqueeze(0)
    valid = torch.ones(1, 15, dtype=torch.bool)
    a = model(tokens, timestamps, valid)["history_states"]
    tokens[:, 10:] += 5
    b = model(tokens, timestamps, valid)["history_states"]
    assert torch.allclose(a[:, :, :10], b[:, :, :10], atol=1e-5)
