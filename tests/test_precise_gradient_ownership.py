import torch

from fate_oia.utils.precise_gradient_ownership import project_target_credit_gradient


def test_target_credit_projection_preserves_grounding_alignment_and_cap():
    grounding = torch.tensor([3.0, 0.0])
    target = torch.tensor([-5.0, 7.0])
    projected = project_target_credit_gradient(grounding, target)
    assert torch.dot(projected, grounding).item() >= -1e-8
    assert projected.norm().item() <= 0.2 * grounding.norm().item() + 1e-7
