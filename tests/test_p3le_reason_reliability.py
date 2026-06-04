import torch

from fate_oia.models.p3le_reason_reliability import ReasonReliabilityHead


def test_reason_reliability_warmup_floor_then_low_floor():
    head = ReasonReliabilityHead(dim=32, reason_dim=21, tail_indices=(5, 6, 9))
    tokens = torch.randn(2, 21, 32)
    logits = torch.randn(2, 21)
    support = torch.randn(2, 21)
    action = torch.rand(2, 4)
    evidence = torch.rand(2, 21)
    early = head(tokens, logits, support, action, evidence, epoch=0, warmup_epochs=5)["reason_reliability"]
    late = head(tokens, logits, support, action, evidence, epoch=10, warmup_epochs=5)["reason_reliability"]
    assert float(early.min()) >= 0.69
    assert float(late.min()) >= 0.049
    assert early.shape == late.shape == (2, 21)
