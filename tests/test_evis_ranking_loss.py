import torch

from fate_oia.losses.ranking_loss import multilabel_hard_negative_ranking_loss


def test_ranking_loss_positive_and_backpropagates():
    logits = torch.tensor([[0.1, 0.9, 0.8, -0.2]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss = multilabel_hard_negative_ranking_loss(logits, labels, margin=0.2, top_k_neg=2)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None
