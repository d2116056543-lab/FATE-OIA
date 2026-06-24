import torch

from fate_oia.losses.acpr_losses import matched_pair_embedding_loss, matched_pair_logit_loss
from fate_oia.models.acpr_pair_memory import ACPRPairMemory


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


def test_pair_memory_ring_buffer_keeps_recent_order_without_cat_rebuild():
    memory = ACPRPairMemory(dim=4, memory_size=3)

    def enqueue(start: int, count: int) -> None:
        names = [f"f{start + i}" for i in range(count)]
        global_embed = torch.arange(start, start + count, dtype=torch.float32).view(count, 1).repeat(1, 4)
        predicate_probs = torch.ones(count, 2)
        action_targets = torch.zeros(count, 4)
        reason_targets = torch.zeros(count, 21)
        contradiction = torch.zeros(count, 21)
        reason_logits = torch.zeros(count, 21)
        reason_emb = torch.zeros(count, 21, 4)
        memory.enqueue(names, global_embed, predicate_probs, action_targets, reason_targets, contradiction, reason_logits, reason_emb)

    enqueue(0, 2)
    enqueue(2, 2)

    ordered = memory._ordered_memory()
    assert ordered["file_names"] == ["f1", "f2", "f3"]
    assert ordered["global_embed"].shape[0] == 3
    assert memory._memory_count == 3


def test_pair_memory_cuda_device_when_available():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory = ACPRPairMemory(dim=4, memory_size=4, memory_device=device)
    names = ["a", "b"]
    global_embed = torch.randn(2, 4)
    predicate_probs = torch.rand(2, 2)
    action_targets = torch.zeros(2, 4)
    reason_targets = torch.zeros(2, 21)
    contradiction = torch.zeros(2, 21)
    reason_logits = torch.zeros(2, 21)
    reason_emb = torch.zeros(2, 21, 4)
    memory.enqueue(names, global_embed, predicate_probs, action_targets, reason_targets, contradiction, reason_logits, reason_emb)
    ordered = memory._ordered_memory()
    assert ordered["global_embed"].device.type == device
