import torch

from fate_oia.models.tida_terminal_innovation import TIDATerminalInnovation


def test_rho_is_stop_gradient_but_predictions_remain_trainable():
    module = TIDATerminalInnovation(dim=8)
    out = module(torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.ones(2, dtype=torch.bool))
    assert not out["innovation_reliability"].requires_grad
    assert out["terminal_prediction_history"].requires_grad
