import torch

from fate_oia.models.dice_atom_reconstructor import straight_through_topk


def test_top2_forward_is_sparse_and_backward_reaches_all_scores():
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]], requires_grad=True)
    weights = straight_through_topk(scores, k=2)
    assert int((weights > 0).sum()) <= 2
    assert torch.allclose(weights.sum(-1), torch.ones(1))
    (weights * torch.tensor([[1.0, 2.0, 4.0, 8.0]])).sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0
