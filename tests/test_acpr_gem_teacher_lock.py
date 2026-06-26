import torch

from fate_oia.utils.acpr_gem_teacher_lock import TeacherBestLock


def test_teacher_best_lock_rejects_without_mutating_state():
    lock = TeacherBestLock(min_delta=1e-4, action_tolerance=0.001, exp_tolerance=0.001)
    theta0 = torch.zeros(25)
    theta1 = torch.ones(25)

    first = lock.maybe_accept(theta0, joint=0.5, action=0.7, exp=0.3, epoch=0)
    rejected = lock.maybe_accept(theta1, joint=0.50001, action=0.6, exp=0.4, epoch=1)

    assert first["accepted"] is True
    assert rejected["accepted"] is False
    assert torch.allclose(lock.best_theta, theta0)
