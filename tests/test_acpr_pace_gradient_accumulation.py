import torch

from fate_oia.utils.acpr_pace_gradient_coordinator import build_gradient_delta, common_descent_gradient


def test_gradient_delta_preserves_non_shared_accumulated_gradients():
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    previous = torch.tensor([0.25, -0.50])
    p.grad = previous.clone()
    action_loss = (p[0] * 2.0 + p[1] * 0.0)
    exp_loss = (-p[0] * 1.0 + p[1] * 0.0)
    deltas, stats = build_gradient_delta(action_loss, exp_loss, [p], max_common_scale=2.0)
    (action_loss + exp_loss).backward()
    for param, delta in deltas:
        param.grad.add_(delta)
    assert torch.isfinite(p.grad).all()
    assert stats["gradient_conflict"] == 1.0
    assert not torch.allclose(p.grad, previous)
    common, _ = common_descent_gradient(torch.tensor([2.0, 0.0]), torch.tensor([-1.0, 0.0]))
    assert torch.dot(common, torch.tensor([2.0, 0.0])) >= -1e-6
    assert torch.dot(common, torch.tensor([-1.0, 0.0])) >= -1e-6
