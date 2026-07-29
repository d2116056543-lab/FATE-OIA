import torch

from fate_oia.losses.meter_counterfactual_losses import dense_factor_intervention_loss


def test_dense_intervention_covers_actions_and_factors() -> None:
    contribution = torch.randn(8, 4, 21) * 0.1
    target = torch.randint(0, 2, (8, 4)).float()
    out = dense_factor_intervention_loss(torch.randn(8, 4), contribution, target)
    assert out["action_coverage"] == 4
    assert out["factor_coverage"] >= 12
    assert torch.isfinite(out["total"])
