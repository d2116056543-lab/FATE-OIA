from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_action_losses import directional_credit_loss


def test_selected_control_directional_credit() -> None:
    selected = torch.tensor([[1.0, -1.0]])
    control = torch.tensor([[0.1, -0.1]])
    direction = torch.tensor([[1.0, -1.0]])
    assert float(directional_credit_loss(selected, control, direction)) < 1e-6
