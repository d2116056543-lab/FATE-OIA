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


def test_ema_evaluation_copy_is_released_before_next_training_epoch():
    source = Path("fate_oia/engine/train_aie_oia.py").read_text(encoding="utf-8")
    assert "ema_model = copy.deepcopy(model).eval()" in source
    assert "ema_checkpoint_state = {" in source
    assert "del ema_model" in source
    assert source.index("del ema_model") < source.index("train_audit_logits")
    assert "best_checkpoint[\"model\"] = ema_checkpoint_state" in source
