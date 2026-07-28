from __future__ import annotations

import torch
from torch import Tensor


def meter_counterfactual_loss(
    selected_effect: Tensor,
    control_effect: Tensor,
    wrong_target_effect: Tensor,
    support_direction: Tensor,
    counter_direction: Tensor,
    *,
    target_action_effect: Tensor | None = None,
    wrong_action_effect: Tensor | None = None,
) -> dict[str, Tensor]:
    """Enforce selected-vs-control and target-specific same-image effects.

    ``selected_effect`` and ``counter_direction`` are signed score changes after
    deleting the selected support/counter evidence.  Optional action effects
    make the loss target-aware without changing the public five-argument API
    used by the unit tests.
    """
    selected_control = torch.relu(0.02 - selected_effect + control_effect).mean()
    specificity_terms = [torch.relu(0.02 - selected_effect + wrong_target_effect)]
    if target_action_effect is not None and wrong_action_effect is not None:
        specificity_terms.append(torch.relu(0.02 - target_action_effect + wrong_action_effect))
    specificity = torch.stack([term.mean() for term in specificity_terms]).mean()
    direction = torch.relu(-support_direction).mean() + torch.relu(-counter_direction).mean()
    total = selected_control + specificity + direction
    return {"selected_control": selected_control, "specificity": specificity, "direction": direction, "total": total}
