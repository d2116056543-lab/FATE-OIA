import torch

from fate_oia.models.mosaic_action_route_policy import compose_final_action_logits


def test_final_action_is_visual_before_admission():
    visual = torch.randn(2, 4)
    shadow = torch.randn(2, 4)
    final = compose_final_action_logits(visual, shadow, torch.zeros(4, dtype=torch.bool))
    assert torch.equal(final, visual)

