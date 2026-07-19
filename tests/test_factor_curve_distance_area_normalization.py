from __future__ import annotations

import math

import torch

from fate_oia.losses.mosaic_icdor_factor_losses import factor_curve_distance_loss


def test_curve_distance_normalizes_mass_error_by_mask_area() -> None:
    prediction = torch.full((1, 1, 2, 3), 0.5)
    target = torch.zeros_like(prediction)
    known = torch.ones((1, 1), dtype=torch.bool)
    loss = factor_curve_distance_loss(prediction, target, known)
    expected = math.log(2.0) + 0.5  # BCE(0.5, 0) + 3 / (2 * 3)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)
