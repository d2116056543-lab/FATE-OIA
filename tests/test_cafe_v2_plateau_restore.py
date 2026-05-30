from __future__ import annotations

import torch

from fate_oia.utils.plateau_rollback import PlateauRestore


def test_plateau_restore_restores_weights_and_decays_lr(tmp_path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = PlateauRestore(patience=0, factor=0.5, min_lr=0.01)
    with torch.no_grad():
        model.weight.fill_(2.0)
    sched.step(1.0, 0, model, opt, None, tmp_path)
    with torch.no_grad():
        model.weight.fill_(0.0)
    event = sched.step(0.0, 1, model, opt, None, tmp_path)
    assert event["restored"]
    assert float(model.weight.item()) == 2.0
    assert opt.param_groups[0]["lr"] == 0.05
    assert (tmp_path / "plateau_restore_event.jsonl").exists()
