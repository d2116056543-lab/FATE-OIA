from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_action_losses import wrong_target_specificity_loss


def test_wrong_target_specificity() -> None:
    selected = torch.tensor([[0.4, 0.4]])
    wrong = torch.tensor([[0.1, 0.1]])
    assert float(wrong_target_specificity_loss(selected, wrong, margin=0.10)) < 1e-6
