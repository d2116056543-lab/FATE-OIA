from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_factor_losses import factor_audit_aligned_losses


def test_factor_audit_loss_alignment() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    masks = torch.sigmoid(torch.randn(4, 3, 5, 5, requires_grad=True))
    outputs = factor_audit_aligned_losses(
        logits, masks, torch.randint(0, 2, (4, 3), dtype=torch.float32), torch.ones(4, 3, dtype=torch.bool)
    )
    required = {"loss_factor_balanced_presence", "loss_factor_query_identity", "loss_factor_image_identity", "loss_factor_prior_gap", "loss_factor_matched_grounding"}
    assert required.issubset(outputs)
    sum(outputs[name] for name in required).backward()
    assert logits.grad is not None
