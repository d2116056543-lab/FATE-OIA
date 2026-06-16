import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_actalign_zero_gate_matches_disabled_final_outputs():
    torch.manual_seed(7)
    base = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, actalign_enabled=False)
    torch.manual_seed(7)
    act = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, actalign_enabled=True, actalign_kwargs={"initial_r2a_gate": 0.0, "initial_pred_gate": 0.0})
    x = torch.randn(2, 3, 360, 640)
    out_base = base(x)
    out_act = act(x)
    assert torch.allclose(out_act["action_logits_final_raw"], out_base["action_logits_final_raw"], atol=1e-6)
    assert torch.allclose(out_act["reason_logits_final_raw"], out_base["reason_logits_final_raw"], atol=1e-6)
    assert "fallback" in out_act["branch_logits"]
    assert "utility" in out_act["branch_logits"]


def test_actalign_gate_does_not_modify_reason_logits():
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, actalign_enabled=True)
    x = torch.randn(1, 3, 360, 640)
    out0 = model(x)
    model.action_utility.set_gates(r2a_gate=torch.ones(4), pred_gate=torch.ones(4))
    out1 = model(x)
    assert torch.allclose(out0["reason_logits_final_raw"], out1["reason_logits_final_raw"], atol=1e-6)
