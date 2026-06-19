from fate_oia.utils.acpr_pace_training_control import PACETrainingControl


def test_training_control_applies_once_after_train_calib_stall():
    ctl = PACETrainingControl(min_epoch=2, patience=2, min_delta=0.01)
    assert not ctl.update(0, 0.50)["cooldown_apply"]
    assert not ctl.update(2, 0.50)["cooldown_apply"]
    assert ctl.update(3, 0.50)["cooldown_apply"]
    assert not ctl.update(4, 0.40)["cooldown_apply"]
