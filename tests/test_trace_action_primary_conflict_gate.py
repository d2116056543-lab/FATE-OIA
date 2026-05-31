import torch
from fate_oia.utils.action_primary_conflict_gate import ActionPrimaryConflictGate


def test_negative_gradient_conflict_downscales_reason():
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    gate = ActionPrimaryConflictGate(conflict_threshold=-0.02, downscale_reason_min=0.35, downscale_evidence_min=0.2)
    action_loss = (p * torch.tensor([1.0, 1.0])).sum()
    reason_loss = -(p * torch.tensor([1.0, 1.0])).sum()
    evidence_loss = reason_loss * 0.5
    stats = gate.compute(action_loss, reason_loss, evidence_loss, [p], epoch=0, latest_act_mf1=None)
    assert stats["grad_cos_action_reason"] < -0.9
    assert stats["applied_reason_scale"] < 1.0
    assert stats["applied_evidence_scale"] < 1.0


def test_positive_gradient_conflict_keeps_scale_one():
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    gate = ActionPrimaryConflictGate(conflict_threshold=-0.02)
    action_loss = (p * torch.tensor([1.0, 1.0])).sum()
    reason_loss = (p * torch.tensor([1.0, 1.0])).sum()
    evidence_loss = reason_loss * 0.5
    stats = gate.compute(action_loss, reason_loss, evidence_loss, [p], epoch=0, latest_act_mf1=None)
    assert stats["grad_cos_action_reason"] > 0.9
    assert stats["applied_reason_scale"] == 1.0
