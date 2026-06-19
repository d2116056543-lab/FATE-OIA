from fate_oia.utils.acpr_teacher_lock import TeacherLockState


def test_teacher_lock_historical_best_accepts_only_improvement():
    t = TeacherLockState()
    assert t.update(1, 0.50, 0.70, 0.40)
    assert not t.update(2, 0.49, 0.71, 0.39)
    assert t.best_epoch == 1
    assert t.update(3, 0.51, 0.70, 0.41)
    assert t.best_epoch == 3
