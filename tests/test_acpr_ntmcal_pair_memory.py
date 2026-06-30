import torch
from fate_oia.models.acpr_ntmcal_pair_memory import NativeTextReasonPairMemory


def test_pair_disabled_before_epoch7():
    m = NativeTextReasonPairMemory()
    logits = torch.randn(3, 21)
    y = torch.zeros(3, 21)
    pu = {"hard_negative_mask": torch.zeros(3, 21)}
    loss, stats = m.loss(logits, y, pu, 0, torch.tensor(1.0))
    assert loss.item() == 0
    assert stats["pair_count_total"] == 0


def test_pair_memory_uses_stored_history_after_epoch7():
    m = NativeTextReasonPairMemory(capacity_per_reason=4)
    logits = torch.tensor([[0.2] * 21, [-0.1] * 21], dtype=torch.float32)
    y = torch.zeros(2, 21)
    y[0, 3] = 1
    pu = {"hard_negative_mask": torch.zeros(2, 21)}
    pu["hard_negative_mask"][1, 3] = 1
    m.enqueue(logits, y, pu)
    loss, stats = m.loss(torch.zeros_like(logits), y, pu, 8, torch.tensor(1.0))
    assert torch.isfinite(loss)
    assert stats["memory_positive_coverage"] > 0
    assert stats["memory_negative_coverage"] > 0
