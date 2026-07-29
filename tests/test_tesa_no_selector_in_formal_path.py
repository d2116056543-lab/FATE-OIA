from fate_oia.models.meter_oia_model import METEROIAModel


def test_no_selector_or_meta_parameters_in_formal_model() -> None:
    names = tuple(dict(METEROIAModel(use_mock_dino=True).named_parameters()))
    assert not any("selector" in name or "meta" in name for name in names)
