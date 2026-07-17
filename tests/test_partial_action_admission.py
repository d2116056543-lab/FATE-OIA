import torch

from fate_oia.models.mosaic_action_route_policy import compose_final_action_logits


def test_partial_action_admission_changes_only_admitted_actions():
    visual = torch.zeros(1, 4)
    shadow = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    final = compose_final_action_logits(visual, shadow, torch.tensor([True, False, True, False]))
    assert torch.equal(final, torch.tensor([[1.0, 0.0, 3.0, 0.0]]))

