from __future__ import annotations

import torch

from fate_oia.losses.mosaic_posterior_ranking import posterior_weighted_reason_ranking_loss
from fate_oia.optim.mosaic_soft_rank_queue import MOSAICSoftRankQueue


def test_posterior_rank_ignores_labels_not_admitted_to_pu() -> None:
    queue = MOSAICSoftRankQueue(capacity=8, label_dim=21)
    queue.enqueue(torch.zeros(2, 21), torch.zeros(2, 21), ["history-a", "history-b"])
    logits = torch.ones(2, 21, requires_grad=True)
    posterior = torch.ones(2, 21)
    gate = torch.zeros(21, dtype=torch.bool)
    loss, stats = posterior_weighted_reason_ranking_loss(
        logits, posterior, ["current-a", "current-b"], queue, label_gate=gate
    )
    assert loss.item() == 0.0
    assert stats["pair_weight_sum"].item() == 0.0
