from types import SimpleNamespace

import torch
from torch import nn

from fate_oia.engine.train_tida_oia import reason_firewall_gradient_audit


class _Geometric(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Linear(1, 1)
        self.reason = nn.Linear(1, 1)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def action_parameters(self):
        return self.action.parameters()

    def reason_parameters(self):
        return self.reason.parameters()


def test_firewall_audit_ignores_disabled_frozen_owner_parameters():
    model = SimpleNamespace(
        action_reader=nn.Linear(1, 1), reason_reader=nn.Linear(1, 1),
        traffic_action=nn.Linear(1, 1), geometric_heads=_Geometric(),
    )
    action_value = model.traffic_action(torch.ones(1, 1)).sum()
    reason_value = model.reason_reader(torch.ones(1, 1)).sum()
    action_names = {
        "action_asl", "action_smooth_ap", "action_base_protect", "action_delta",
        "action_flow_credit", "action_flow_no_harm", "action_utility_calibration",
        "geometric_action_aux", "geometric_action_rank", "geometric_action_prefix", "geometric_action_delta",
        "traffic_action_aux", "traffic_action_rank", "traffic_action_delta",
    }
    reason_names = {
        "reason_partial", "reason_rank", "reason_soft_f1", "reason_delta",
        "reason_flow_credit", "reason_flow_no_harm", "reason_positive_no_harm",
        "reason_utility_calibration", "geometric_reason_aux", "geometric_reason_rank",
        "geometric_reason_prefix", "geometric_reason_delta",
    }
    rows = {
        name: SimpleNamespace(weight=1.0, value=action_value if name in action_names else reason_value)
        for name in action_names | reason_names
    }
    result = reason_firewall_gradient_audit(SimpleNamespace(rows=rows), model)
    assert result == {"reason_loss_to_action_owner": 0.0, "action_loss_to_reason_owner": 0.0}
