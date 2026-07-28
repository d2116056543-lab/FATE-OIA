import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_meter_model_has_full_formal_path_and_reason_firewall() -> None:
    torch.manual_seed(9)
    model = METEROIAModel(use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)
    initial = model(images, progress=0.0)
    full = model(images, progress=1.0)

    assert initial["action_logits_final"].shape == (1, 4)
    assert initial["reason_logits_final"].shape == (1, 21)
    assert full["factor_support_map"].shape == (1, 21, 3600)
    torch.testing.assert_close(initial["action_logits_final"], initial["action_logits_calalign"], atol=1e-6, rtol=0)
    torch.testing.assert_close(initial["reason_logits_final"], initial["reason_logits_calalign"], atol=1e-6, rtol=0)

    action_before = full["action_logits_final"].detach().clone()
    with torch.no_grad():
        for parameter in model.reason_decoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    action_after = model(images, progress=1.0)["action_logits_final"]
    torch.testing.assert_close(action_before, action_after, atol=1e-6, rtol=0)
