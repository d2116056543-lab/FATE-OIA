import torch

from fate_oia.utils.acpr_action_gradient_guard import ACPRActionGradientGuard


def test_gradient_guard_projects_conflicting_gradient():
    c = torch.tensor([-1.0, 0.0])
    a = torch.tensor([1.0, 0.0])
    p = ACPRActionGradientGuard.project(c, a)
    assert torch.dot(p.flatten(), a.flatten()).item() >= -1e-7


def test_gradient_guard_leaves_aligned_gradient():
    c = torch.tensor([2.0, 1.0])
    a = torch.tensor([1.0, 0.0])
    p = ACPRActionGradientGuard.project(c, a)
    assert torch.allclose(p, c)


def test_gradient_guard_projects_model_param_grad_and_logs():
    guard = ACPRActionGradientGuard(mode="log_then_project", project_after_epoch=0, every_n_steps=1)
    p = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    p.grad = torch.tensor([-1.0, 0.0])
    action_grads = [torch.tensor([1.0, 0.0])]
    stats = guard.project_model_grads([("p", p)], action_grads, epoch=0, step=1)
    assert torch.dot(p.grad, action_grads[0]).item() >= -1e-7
    assert stats["grad_conflict"] is True
    assert stats["conflict_param_count"] == 1
