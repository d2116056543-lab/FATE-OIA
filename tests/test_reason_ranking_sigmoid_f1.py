from __future__ import annotations

import torch

from fate_oia.losses.reason_ranking_loss import reason_pairwise_ranking_loss
from fate_oia.losses.sigmoid_f1_loss import sigmoid_macro_f1_loss


def test_reason_pairwise_ranking_loss_backpropagates() -> None:
    logits = torch.tensor([[0.4, -0.2, 0.1], [-0.3, 0.8, -0.1]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss = reason_pairwise_ranking_loss(logits, labels, label_indices=[0, 1, 2], margin=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_sigmoid_macro_f1_loss_is_low_for_good_logits() -> None:
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    good = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    bad = torch.zeros_like(good)
    assert sigmoid_macro_f1_loss(good, labels) < sigmoid_macro_f1_loss(bad, labels)
