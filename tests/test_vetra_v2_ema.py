import torch
from torch import nn
from pathlib import Path

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


def test_average_parameters_swaps_and_restores_after_exception():
    model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
        for buffer in model.buffers():
            if torch.is_floating_point(buffer):
                buffer.add_(3.0)
    online = {name: value.detach().clone() for name, value in model.state_dict().items()}
    average = ema.state_dict()

    try:
        with ema.average_parameters(model):
            for name, value in model.state_dict().items():
                assert torch.equal(value, average[name])
            raise RuntimeError("exercise restoration")
    except RuntimeError as error:
        assert str(error) == "exercise restoration"

    for name, value in model.state_dict().items():
        assert torch.equal(value, online[name])
    restored_average = ema.state_dict()
    for name, value in average.items():
        assert torch.equal(restored_average[name], value)


def test_ema_evaluation_swaps_without_deepcopy():
    source = Path("fate_oia/engine/train_aie_oia.py").read_text(encoding="utf-8")
    assert "with ema.average_parameters(model):" in source
    assert "copy.deepcopy(model)" not in source
    assert "ema_checkpoint_state = {" in source
    assert "best_checkpoint[\"model\"] = ema_checkpoint_state" in source
