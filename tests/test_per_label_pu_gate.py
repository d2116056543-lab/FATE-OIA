from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_reason_losses import build_per_label_pu_gate


def test_per_label_pu_gate() -> None:
    margins = torch.full((21,), -0.01)
    margins[[0, 2]] = torch.tensor([0.01, 0.02])
    gate = build_per_label_pu_gate(margins, minimum_margin=0.0)
    assert torch.equal(gate[:3], torch.tensor([True, False, True]))
