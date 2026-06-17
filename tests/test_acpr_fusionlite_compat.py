import torch
from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_fusionlite_model_initial_compatibility_and_deploy():
    torch.manual_seed(2)
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, use_fusionlite=True)
    out = model(torch.randn(2, 3, 360, 640))
    assert torch.allclose(out["action_logits_base"], out["action_logits_direct_legacy"], atol=1e-6)
    assert torch.allclose(out["action_logits_fusionlite"], out["action_logits_direct_legacy"], atol=1e-6)
    assert out["reason_logits_base"].shape == (2, 21)
    theta = out["threshold_logit"].view(1, -1)
    assert torch.allclose(out["logits_deploy"], out["logits_base_fixed"] - theta, atol=1e-6)
    assert "action_set_logits" in out and out["action_set_logits"].shape == (2, 16)
