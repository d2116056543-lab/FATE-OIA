import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_cheap_ablation_reuses_one_encoded_field() -> None:
    model = METEROIAModel(dim=384, use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)
    field = model.encode_images(images)
    before = model._encode_call_count
    clean = model.decode_from_field(field, progress=1.0)
    uniform = model.decode_from_field(field, progress=1.0, diagnostic_modes=("state_uniform",))
    factor_off = model.decode_from_field(field, progress=1.0, diagnostic_modes=("factor_off",))
    assert model._encode_call_count == before == 1
    assert not torch.allclose(clean["action_factor_values"], uniform["action_factor_values"])
    assert torch.equal(factor_off["action_logits_final"], factor_off["action_logits_visual"])

