import torch

from fate_oia.models.tida_predicate_differential import TIDAPredicateDifferential, robust_common_motion


def test_static_translation_is_removed_from_relative_velocity():
    velocity = torch.ones(1, 2, 4, 3) * 2.0
    static_mask = torch.tensor([True, True, False, False])
    common = robust_common_motion(velocity, static_mask)
    relative = velocity - common[:, :, None]
    assert common.requires_grad is False
    torch.testing.assert_close(relative[:, :, :2], torch.zeros_like(relative[:, :, :2]), atol=1e-6, rtol=0)


def test_predicate_specific_common_motion_projection_starts_as_identity():
    roles = {"static_anchor": ["p0"], "dynamic_actor": ["p1"], "terminal_context": []}
    module = TIDAPredicateDifferential(dim=3, predicate_names=["p0", "p1"], roles=roles)
    scale = 1.0 + 0.25 * torch.tanh(module.common_projection_raw)
    torch.testing.assert_close(scale, torch.ones_like(scale))
    assert scale.shape == (2, 3)
