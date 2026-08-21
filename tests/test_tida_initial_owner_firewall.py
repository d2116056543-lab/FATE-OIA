import torch
from torch import nn

from fate_oia.engine.train_tida_oia import _apply_initial_owner_firewall


class _Owners(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Parameter(torch.ones(1))
        self.reason = nn.Parameter(torch.ones(1))
        self.differential = nn.Parameter(torch.ones(1))

    def owner_parameters(self):
        return {"temporal_action": [self.action], "temporal_reason": [self.reason], "predicate_differential": [self.differential]}


def test_initial_window_blocks_action_reason_but_keeps_differential_gradient():
    model = _Owners()
    (model.action + model.reason + model.differential).backward()
    _apply_initial_owner_firewall(model, 0.0)
    assert model.action.grad is None and model.reason.grad is None
    assert model.differential.grad is not None
