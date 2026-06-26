from __future__ import annotations

import torch

from fate_oia.losses import acpr_losses as L


def test_partial_reason_loss_accepts_unknown_negative_weight():
    logits = torch.zeros(2, 21)
    target = torch.zeros(2, 21)
    contradiction = torch.zeros(2, 21)
    loss = L.partial_label_reason_loss(logits, target, contradiction)
    assert torch.isfinite(loss)

