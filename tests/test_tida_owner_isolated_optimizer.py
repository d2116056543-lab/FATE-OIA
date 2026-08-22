import torch
from torch import nn

from fate_oia.engine.train_tida_oia import build_optimizer


class _OwnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action = nn.Linear(2, 1)
        self.reason = nn.Linear(2, 1)

    def owner_parameters(self):
        return {
            "temporal_action": list(self.action.parameters()),
            "temporal_reason": list(self.reason.parameters()),
        }


def test_owner_isolated_optimizer_uses_zero_lr_for_unselected_owner():
    model = _OwnerModel()
    config = {
        "training": {
            "lr": {"temporal_action": 0.2, "temporal_reason": 0.1},
            "weight_decay": 0.05,
        }
    }
    optimizer = build_optimizer(model, config, train_owners={"temporal_reason"})
    by_name = {group["name"]: group for group in optimizer.param_groups}
    assert by_name["temporal_action"]["lr"] == 0.0
    assert by_name["temporal_reason"]["lr"] == 0.1
    before_action = model.action.weight.detach().clone()
    before_reason = model.reason.weight.detach().clone()
    (model.action(torch.ones(1, 2)).sum() + model.reason(torch.ones(1, 2)).sum()).backward()
    optimizer.step()
    torch.testing.assert_close(model.action.weight, before_action)
    assert not torch.equal(model.reason.weight, before_reason)
