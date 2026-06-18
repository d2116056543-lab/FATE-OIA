import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_seca_zero_gate_matches_legacy_action_base():
    torch.manual_seed(1)
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, seca_enabled=True)
    out = model(torch.randn(2, 3, 360, 640))
    assert torch.allclose(out["action_logits_base"], out["action_logits_legacy_base"], atol=1e-6)
    assert torch.allclose(out["logits_base_fixed"], out["logits_legacy_base_fixed"], atol=1e-6)
    assert torch.allclose(out["branch_logits"]["legacy_base_fixed"], out["logits_legacy_base_fixed"])
    assert torch.allclose(out["branch_logits"]["seca_base_fixed"], out["logits_base_fixed"])
