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
