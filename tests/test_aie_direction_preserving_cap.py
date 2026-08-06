import torch

from fate_oia.models.aie_contribution_head import direction_preserving_l2_cap


def test_l2_cap_preserves_direction_and_ranking():
    logits = torch.tensor([[30.0, 20.0, -10.0, 5.0]])
    capped = direction_preserving_l2_cap(logits, 20.0)
    assert float(capped.norm()) <= 20.00001
    torch.testing.assert_close(capped / capped.norm(), logits / logits.norm())
    assert torch.equal(capped.argsort(), logits.argsort())

