import torch

from fate_oia.losses.acpr_losses import matched_pair_embedding_loss, matched_pair_logit_loss


def test_memory_negative_logit_uses_detached_logit_not_batch_index():
    logits = torch.zeros(2, 21, requires_grad=True)
    logits.data[0, 7] = 0.0
    pairs = {
        "pair_pos_indices": torch.tensor([0]),
        "pair_neg_indices": torch.tensor([-1]),
        "pair_neg_memory_indices": torch.tensor([7]),
        "pair_reason_ids": torch.tensor([7]),
        "pair_weights": torch.tensor([1.0]),
        "pair_neg_logits_detached": torch.tensor([1.0]),
        "pair_neg_is_memory": torch.tensor([True]),
        "pair_active_mask": torch.tensor([True]),
    }
    loss = matched_pair_logit_loss(logits, pairs)
    loss.backward()
    assert float(loss.detach()) > 0
    assert logits.grad is not None
    assert logits.grad[0, 7] < 0
    assert torch.count_nonzero(logits.grad[:, [i for i in range(21) if i != 7]]) == 0


def test_memory_negative_embedding_uses_reason_specific_detached_embedding():
    embeddings = torch.randn(2, 21, 8, requires_grad=True)
    neg = embeddings.detach()[0, 3].clone().view(1, 8)
    neg[:, 0] += 0.5
    pairs = {
        "pair_pos_indices": torch.tensor([0]),
        "pair_neg_indices": torch.tensor([-1]),
        "pair_neg_memory_indices": torch.tensor([9]),
        "pair_reason_ids": torch.tensor([3]),
        "pair_weights": torch.tensor([1.0]),
        "pair_neg_embedding_detached": neg,
        "pair_neg_is_memory": torch.tensor([True]),
        "pair_active_mask": torch.tensor([True]),
    }
    loss = matched_pair_embedding_loss(embeddings, pairs)
    loss.backward()
    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    touched = embeddings.grad.abs().sum(dim=-1) > 0
    assert bool(touched[0, 3])
    assert int(touched.sum()) == 1


def test_logit_pair_loss_caps_extreme_hinge_but_reports_raw_stats():
    logits = torch.tensor([[0.0], [100.0]], requires_grad=True)
    pairs = {
        "pair_pos_indices": torch.tensor([0]),
        "pair_neg_indices": torch.tensor([1]),
        "pair_reason_ids": torch.tensor([0]),
        "pair_weights": torch.tensor([1.0]),
        "pair_active_mask": torch.tensor([True]),
    }
    loss, stats = matched_pair_logit_loss(logits, pairs, margin=0.25, max_hinge=4.0, return_stats=True)
    assert torch.allclose(loss.detach(), torch.tensor(4.0))
    assert stats["hinge_mean"] > 100.0
