import torch

from fate_oia.models.tida_temporal_encoder import TIDATemporalEncoder


def test_all_invalid_history_returns_zero_summary():
    model = TIDATemporalEncoder(dim=8, num_layers=1, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 14, 3, 8)
    ts = torch.linspace(-5, 0, 15).repeat(2, 1)
    valid = torch.zeros(2, 15, dtype=torch.bool); valid[:, -1] = True
    out = model(x, ts, valid)
    assert torch.equal(out["history_summary"], torch.zeros_like(out["history_summary"]))
