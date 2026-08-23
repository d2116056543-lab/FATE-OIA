import torch

from fate_oia.losses.tida_losses import target_conditioned_geometric_correction_loss


def test_target_conditioned_correction_rewards_gt_margin_and_has_finite_gradient():
    base = torch.tensor([[-0.2, 0.2], [0.1, -0.1]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    good = torch.tensor([[0.3, -0.3], [0.3, -0.3]], requires_grad=True)
    bad = -good.detach()
    motion = torch.ones(2, 4) * 0.1
    good_loss = target_conditioned_geometric_correction_loss(base, good, target, motion)
    bad_loss = target_conditioned_geometric_correction_loss(base, bad, target, motion)
    assert good_loss < bad_loss
    good_loss.backward()
    assert torch.isfinite(good.grad).all()


def test_confident_correct_base_is_protected_from_harmful_delta():
    base = torch.tensor([[2.0, -2.0]])
    target = torch.tensor([[1.0, 0.0]])
    motion = torch.ones(1, 3)
    safe = target_conditioned_geometric_correction_loss(base, torch.zeros_like(base), target, motion)
    harmful = target_conditioned_geometric_correction_loss(base, torch.tensor([[-1.0, 1.0]]), target, motion)
    assert harmful > safe
