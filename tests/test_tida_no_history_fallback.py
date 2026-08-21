import torch

from fate_oia.models.tida_terminal_innovation import TIDATerminalInnovation


def test_invalid_history_forces_zero_rho_and_xi():
    module = TIDATerminalInnovation(dim=8)
    out = module(torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.zeros(2, dtype=torch.bool))
    assert torch.equal(out["innovation_reliability"], torch.zeros_like(out["innovation_reliability"]))
    assert torch.equal(out["innovation_token"], torch.zeros_like(out["innovation_token"]))
