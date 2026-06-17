import torch
from fate_oia.models.acpr_oia_model import ACPROIAModel


def test_pmt_forward_mock_shapes_and_zero_init():
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, pmt_kwargs={"enabled": True})
    image = torch.randn(2, 3, 360, 640)
    out = model(image, epoch=0)
    assert out["action_logits_final_raw"].shape == (2, 4)
    assert out["reason_logits_final_raw"].shape == (2, 21)
    assert out["triadic_action_delta"].abs().max() < 1e-6
    assert "deploy_fixed_pmt" in out["branch_logits"]


def test_pmt_disabled_path_runs():
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, pmt_kwargs={"enabled": False})
    out = model(torch.randn(1, 3, 360, 640), epoch=0)
    assert out["action_logits_final_raw"].shape == (1, 4)
