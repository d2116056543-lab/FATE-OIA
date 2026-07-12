from __future__ import annotations

import math

import torch

from fate_oia.engine.diagnose_mosaic_factor_transport import binary_roc_auc


def test_binary_roc_auc_handles_perfect_and_reversed_rankings() -> None:
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert binary_roc_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), targets) == 1.0
    assert binary_roc_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), targets) == 0.0


def test_binary_roc_auc_is_nan_without_both_classes() -> None:
    value = binary_roc_auc(torch.tensor([0.1, 0.2]), torch.zeros(2))
    assert math.isnan(value)
