import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_heca_state_off_recomputes_measurement_and_action_bridge_paths():
    torch.manual_seed(9)
    model = METEROIAModel(dim=384, use_mock_dino=True).eval()
    images = torch.randn(2, 3, 360, 640)
    field = model.encode_images(images)
    clean = model.decode_from_field(field, progress=1.0)
    state_off = model.decode_from_field(
        field, progress=1.0, diagnostic_modes=("state_off",)
    )
    assert not torch.equal(
        clean["factor_measurement_token"], state_off["factor_measurement_token"]
    )
    assert not torch.equal(
        clean["factor_action_bridge_token"], state_off["factor_action_bridge_token"]
    )
