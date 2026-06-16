import torch

from fate_oia.utils.acpr_action_pareto_gate import ActionParetoGate


def test_action_pareto_gate_opens_only_improved_supported_label():
    g = ActionParetoGate(action_dim=4, gate_ema=1.0, action_margin=0.001, min_support=2)
    labels = torch.zeros(6, 4)
    labels[:3, 2] = 1
    base = torch.zeros(6, 4)
    r2a = base.clone(); r2a[:3, 2] = 4; r2a[3:, 2] = -4
    pred = base.clone()
    stats = g.update(base, r2a, pred, labels)
    assert g.r2a_gate.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert stats["source"] == "train_calib_only"


def test_action_pareto_gate_low_support_stays_closed():
    g = ActionParetoGate(action_dim=4, gate_ema=1.0, min_support=5)
    labels = torch.zeros(4, 4); labels[0, 1] = 1
    base = torch.zeros(4, 4)
    cand = base.clone(); cand[0, 1] = 5
    g.update(base, cand, cand, labels)
    assert g.r2a_gate[1].item() == 0.0
