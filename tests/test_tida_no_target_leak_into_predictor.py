import torch

from fate_oia.models.tida_terminal_innovation import TIDATerminalInnovation


def test_terminal_target_does_not_change_predictor_outputs():
    module = TIDATerminalInnovation(dim=8).eval()
    static = torch.randn(2, 4, 8)
    history = torch.randn(2, 4, 8)
    a = module(static, history, torch.randn(2, 4, 8), torch.ones(2, dtype=torch.bool))
    b = module(static, history, torch.randn(2, 4, 8) * 100, torch.ones(2, dtype=torch.bool))
    assert torch.equal(a["terminal_prediction_history"], b["terminal_prediction_history"])
    assert torch.equal(a["terminal_prediction_no_history"], b["terminal_prediction_no_history"])
