import torch

from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_calalign_forward_exposes_base_deploy_and_uses_deploy_as_final_raw():
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True)
    out = model(torch.randn(2, 3, 360, 640))

    assert out["action_logits_base"].shape == (2, 4)
    assert out["reason_logits_base"].shape == (2, 21)
    assert out["action_logits_deploy"].shape == (2, 4)
    assert out["reason_logits_deploy"].shape == (2, 21)
    assert out["threshold_logit"].shape == (25,)
    assert torch.allclose(out["action_logits_final_raw"], out["action_logits_deploy"])
    assert torch.allclose(out["reason_logits_final_raw"], out["reason_logits_deploy"])
    assert torch.allclose(out["branch_logits"]["base_fixed"], out["logits_base_fixed"])
    assert torch.allclose(out["branch_logits"]["deploy_fixed"], out["logits_final_raw"])


def test_acpr_forward_can_disable_threshold_for_old_config_behavior():
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=False)
    out = model(torch.randn(1, 3, 360, 640))

    assert torch.allclose(out["action_logits_final_raw"], out["action_logits_base"])
    assert torch.allclose(out["reason_logits_final_raw"], out["reason_logits_base"])
