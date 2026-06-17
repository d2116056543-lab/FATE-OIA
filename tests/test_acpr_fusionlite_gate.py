import torch
from fate_oia.models.acpr_fusionlite_gate import ACPRFusionLiteGate


def test_fusionlite_zero_init_equivalence_and_bounds():
    torch.manual_seed(1)
    gate = ACPRFusionLiteGate(dim=16, num_predicates=5)
    b = 3
    action_nodes = torch.randn(b, 4, 16)
    reason_nodes = torch.randn(b, 21, 16)
    pred = torch.rand(b, 5)
    av = torch.randn(b, 4)
    ar = torch.randn(b, 4)
    old = torch.rand(b, 4).clamp(0.10, 0.90)
    arm = torch.rand(4, 21)
    arm = arm / arm.sum(1, keepdim=True)
    apm = torch.rand(4, 5)
    apm = apm / apm.sum(1, keepdim=True)
    out = gate(action_nodes, reason_nodes, pred, av, ar, old, arm, apm)
    expected = old * av + (1 - old) * ar
    assert torch.allclose(out["action_logits_fusionlite"], expected, atol=1e-6)
    with torch.no_grad():
        gate.delta_mlp[-1].bias.fill_(10.0)
    out2 = gate(action_nodes, reason_nodes, pred, av, ar, old, arm, apm)
    assert out2["fusionlite_delta_gate"].abs().max() <= gate.max_delta + 1e-6
    assert out2["fusionlite_gate"].min() >= 0.10 - 1e-6
    assert out2["fusionlite_gate"].max() <= 0.90 + 1e-6
