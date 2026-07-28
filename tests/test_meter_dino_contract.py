from fate_oia.models.meter_calalign_foundation import METERCalAlignFoundation


def test_dino_is_frozen_and_has_required_taps() -> None:
    model = METERCalAlignFoundation(use_mock_dino=True)
    assert tuple(model.selected_layers) == (3, 7, 11)
    assert all(not parameter.requires_grad for parameter in model.dino.parameters())
    assert model.dino.training is False
