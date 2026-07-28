import torch

from fate_oia.losses.meter_counterfactual_losses import meter_counterfactual_loss


def test_counterfactual_loss_has_selected_control_and_target_terms() -> None:
    values = [torch.tensor([0.2, 0.3]), torch.tensor([0.0, 0.1]), torch.tensor([-0.1, 0.0])]
    result = meter_counterfactual_loss(*values, values[0], values[0], target_action_effect=values[0], wrong_action_effect=values[1])
    assert {"selected_control", "specificity", "direction", "total"} <= result.keys()
    assert torch.isfinite(result["total"])
