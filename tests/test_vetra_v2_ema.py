import torch
from torch import nn

from fate_oia.utils.aie_ema import ModelEMA


def test_ema_updates_and_restores_state():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1.0)
    ema = ModelEMA(model, decay=0.5)
    model.weight.data.fill_(3.0)
    ema.update(model)
    torch.testing.assert_close(ema.state_dict()["weight"], torch.full_like(model.weight, 2.0))

    target = nn.Linear(2, 1, bias=False)
    ema.copy_to(target)
    torch.testing.assert_close(target.weight, torch.full_like(target.weight, 2.0))
