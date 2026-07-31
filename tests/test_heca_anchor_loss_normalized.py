import math

import torch

from fate_oia.losses.meter_grounding_losses import source_weighted_anchor_loss


def test_anchor_nll_is_normalized_by_log_valid_partition() -> None:
    n = 3600
    predicted = torch.full((1, 1, n), 1.0 / n)
    target = torch.zeros_like(predicted)
    target[..., 0] = 1
    nll, _ = source_weighted_anchor_loss(
        predicted, target, torch.ones(1, 1), torch.ones(1, 1)
    )
    torch.testing.assert_close(nll, torch.tensor(1.0), atol=1e-4, rtol=1e-4)
    assert math.isfinite(float(nll))

