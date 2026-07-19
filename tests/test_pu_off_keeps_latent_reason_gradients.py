from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_reason_losses import latent_reason_core_loss


def test_pu_off_keeps_latent_reason_gradients() -> None:
    logits = torch.zeros(4, 21, requires_grad=True)
    targets = torch.randint(0, 2, (4, 21), dtype=torch.float32)
    loss = latent_reason_core_loss(logits, targets, pu_enabled=False)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0
