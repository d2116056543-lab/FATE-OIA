from __future__ import annotations

import torch

from fate_oia.models.mosaic_action_route_policy import compose_final_action_logits


def test_partial_action_admission() -> None:
    visual = torch.zeros(1, 4)
    shadow = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    final = compose_final_action_logits(visual, shadow, torch.tensor([True, False, True, False]))
    assert torch.equal(final, torch.tensor([[0.5, 0.0, 0.5, 0.0]]))
