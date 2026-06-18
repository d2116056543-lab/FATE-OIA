import torch
from fate_oia.utils.acpr_seca_training_control import ACPRSECATrainingControl, update_warmup_cosine_multiplier


def test_seca_scheduler_and_cooldown():
    assert update_warmup_cosine_multiplier(0, 100, 10, 0.05) >= 0.05
    c = ACPRSECATrainingControl(cooldown_start_epoch=6, patience=2)
    assert not c.update(6, 0.5)["cooldown_active"]
    assert not c.update(7, 0.4)["cooldown_active"]
    assert c.update(8, 0.3)["cooldown_active"]


from fate_oia.utils.acpr_seca_training_control import apply_lr_cooldown


def test_apply_lr_cooldown_respects_threshold_group():
    param = torch.nn.Parameter(torch.tensor(1.0))
    param2 = torch.nn.Parameter(torch.tensor(2.0))
    opt = torch.optim.SGD([
        {"params": [param], "lr": 0.2, "name": "trunk_without_seca"},
        {"params": [param2], "lr": 0.7, "name": "threshold"},
    ])
    for group in opt.param_groups:
        group["base_lr"] = group["lr"]
    apply_lr_cooldown(opt, {"non_threshold_lr_mult": 0.2, "threshold_lr_mult": 0.5}, lr_multiplier=0.5)
    assert opt.param_groups[0]["lr"] == 0.020000000000000004
    assert opt.param_groups[1]["lr"] == 0.175
