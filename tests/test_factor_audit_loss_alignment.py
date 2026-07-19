from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_factor_losses import (
    factor_audit_aligned_losses,
    factor_query_identity_loss,
)


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


def test_query_identity_handles_multiple_active_rows_and_same_type_negatives() -> None:
    features = torch.randn(5, 4, 8, requires_grad=True)
    queries = torch.randn(4, 8)
    type_ids = torch.tensor([0, 0, 0, 1])
    presence = torch.ones(5, 4, dtype=torch.bool)
    known = torch.ones(5, 4, dtype=torch.bool)

    loss = factor_query_identity_loss(
        features,
        factor_queries=queries,
        factor_type_ids=type_ids,
        presence_targets=presence,
        presence_known_mask=known,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
