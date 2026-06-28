from __future__ import annotations

import torch


def test_certified_pair_loss_zero_when_no_reliable_negatives():
    from fate_oia.losses.pmcal_certified_pair_loss import certified_near_boundary_pair_loss
    logits = torch.zeros(2, 21)
    labels = torch.zeros(2, 21)
    state = {
        "positive_mask": labels,
        "reliable_negative_mask": torch.zeros_like(labels),
    }
    loss, stats = certified_near_boundary_pair_loss(logits, labels, state)
    assert loss.item() == 0
    assert stats["certified_pair_count"] == 0
