import torch

from fate_oia.models.acpr_semantic_evidence_coattention import ACPRSparseEvidenceCoAttention


def test_seca_shapes_null_token_and_zero_gate_identity():
    m = ACPRSparseEvidenceCoAttention(dim=32, num_heads=4)
    action = torch.randn(2, 4, 32)
    reason = torch.randn(2, 21, 32)
    out = m(action, reason)
    assert out["action_nodes_seca"].shape == action.shape
    assert out["action_reason_attention_heads"].shape == (2, 4, 4, 22)
    assert out["action_reason_attention"].shape == (2, 4, 22)
    assert out["action_reason_attention_no_null"].shape == (2, 4, 21)
    assert out["null_attention"].shape == (2, 4)
    assert torch.allclose(out["residual_scale"], torch.zeros(4), atol=1e-8)
    assert torch.allclose(out["action_nodes_seca"], action, atol=1e-6)


def test_seca_non_dead_gradient_after_gate_step():
    torch.manual_seed(0)
    m = ACPRSparseEvidenceCoAttention(dim=32, num_heads=4)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    action = torch.randn(2, 4, 32, requires_grad=True)
    reason = torch.randn(2, 21, 32, requires_grad=True)
    out = m(action, reason)
    loss = out["action_nodes_seca"].pow(2).mean()
    loss.backward()
    assert m.residual_gate_raw.grad is not None
    assert torch.isfinite(m.residual_gate_raw.grad).all()
    assert m.residual_gate_raw.grad.abs().sum() > 0
    opt.step(); opt.zero_grad(set_to_none=True)
    out2 = m(action, reason)
    out2["action_nodes_seca"].pow(2).mean().backward()
    grads = [m.q_proj.weight.grad, m.k_proj.weight.grad, m.v_proj.weight.grad, m.out_proj.weight.grad]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
