import torch
from torch import nn

from fate_oia.models.tida_terminal_innovation import TIDATerminalInnovation


class _HistoryIdentity(nn.Module):
    def forward(self, _static, history):
        return history


def test_repeated_last_like_null_history_has_lower_rho_than_informative_history():
    module = TIDATerminalInnovation(dim=4)
    module.history_predictor = _HistoryIdentity()
    static = torch.zeros(1, 1, 4)
    target = torch.tensor([[[1.0, -1.0, 0.5, -0.5]]])
    valid = torch.ones(1, dtype=torch.bool)
    real = module(static, target, target, valid)["innovation_reliability"]
    repeated = module(static, torch.zeros_like(target), target, valid)["innovation_reliability"]
    assert real.mean() > repeated.mean()
