import torch
from fate_oia.losses.acpr_pmt_losses import (
    triadic_consistency_loss,
    predicate_conditioned_pu_reason_loss,
    capped_pair_loss,
)


def test_pu_positive_weight_and_variable_negative_weight():
    logits = torch.zeros(1, 3)
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    contradiction = torch.tensor([[0.0, 0.0, 1.0]])
    loss, stats = predicate_conditioned_pu_reason_loss(logits, targets, contradiction, neg_min=0.2)
    assert torch.isfinite(loss)
    assert stats["positive_weight_mean"] == 1.0
    assert stats["negative_weight_mean"] > 0.2


def test_triadic_consistency_positive_only_and_finite():
    support = torch.rand(2, 4, 21)
    action = torch.randint(0, 2, (2, 4)).float()
    reason = torch.randint(0, 2, (2, 21)).float()
    loss, stats = triadic_consistency_loss(support, action, reason, None)
    assert torch.isfinite(loss)
    assert "positive_count" in stats


def test_pair_cap_clips_large_pair_loss():
    raw = torch.tensor(10.0)
    main = torch.tensor(2.0)
    capped, stats = capped_pair_loss(raw, main, cap_ratio=0.1)
    assert float(capped) <= 0.20001
    assert stats["pair_cap_active_rate"] == 1.0
