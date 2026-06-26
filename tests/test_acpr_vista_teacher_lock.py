from __future__ import annotations

from fate_oia.utils.acpr_teacher_lock import ACPRTeacherLockState, should_accept_teacher


def test_teacher_lock_rejects_lower_joint_candidate():
    state = ACPRTeacherLockState(best_joint=0.50, best_action=0.70, best_exp=0.40, best_epoch=1)
    assert not should_accept_teacher(state, 0.49, 0.71, 0.39)
    assert should_accept_teacher(state, 0.51, 0.70, 0.41)

