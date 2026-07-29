import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_tesa_state_off_changes_the_formal_typed_token_path():
    torch.manual_seed(9)
    model = METEROIAModel(dim=32, use_mock_dino=True).eval()
    images = torch.randn(2, 3, 360, 640)
    field = model.encode_images(images)
    clean = model.decode_from_field(field, progress=1.0)
    state_off = model.decode_from_field(
        field, progress=1.0, diagnostic_modes=("state_off",)
    )
    assert not torch.equal(
        clean["factor_typed_token"], state_off["factor_typed_token"]
    )
    assert not torch.equal(
        clean["action_logits_final"], state_off["action_logits_final"]
    )
