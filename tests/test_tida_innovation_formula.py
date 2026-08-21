import torch

from fate_oia.models.tida_terminal_innovation import TIDATerminalInnovation


def test_rho_and_innovation_recompute_exactly():
    module = TIDATerminalInnovation(dim=8).eval()
    static = torch.randn(2, 3, 8)
    history = torch.randn(2, 3, 8)
    target = torch.randn(2, 3, 8)
    valid = torch.ones(2, dtype=torch.bool)
    out = module(static, history, target, valid)
    expected = ((out["terminal_error_no_history"] - out["terminal_error_history"]) / (out["terminal_error_no_history"] + module.eps)).clamp(0, 1)
    assert torch.allclose(out["innovation_reliability"], expected, atol=1e-6)
    assert not out["innovation_reliability"].requires_grad
