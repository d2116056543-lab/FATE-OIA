import torch

from fate_oia.models.acpr_semantic_evidence_coattention import ACPRSparseEvidenceCoAttention


def test_reason_gradient_is_scaled_through_action_path():
    torch.manual_seed(2)
    m = ACPRSparseEvidenceCoAttention(dim=16, num_heads=4, evidence_grad_scale=0.25)
    with torch.no_grad():
        m.residual_gate_raw.fill_(0.5)
    action = torch.randn(1, 4, 16, requires_grad=True)
    reason = torch.randn(1, 21, 16, requires_grad=True)
    out = m(action, reason)
    loss = out["action_nodes_seca"].sum()
    loss.backward()
    assert reason.grad is not None
    assert torch.isfinite(reason.grad).all()
    assert reason.grad.abs().sum() > 0
