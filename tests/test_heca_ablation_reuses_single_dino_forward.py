import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_cheap_ablation_reuses_single_dino_forward() -> None:
    model = METEROIAModel(dim=32, use_mock_dino=True)
    image = torch.randn(1, 3, 360, 640)
    field = model.encode_images(image)
    count = model._encode_call_count
    for modes in ((), ("factor_off",), ("state_uniform",), ("reason_correction_off",)):
        output = model.decode_from_field(field, progress=0.4, diagnostic_modes=modes)
        assert "action_logits_visual" in output and "reason_logits_global" in output
    assert model._encode_call_count == count
