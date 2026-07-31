import inspect

from fate_oia.models.meter_oia_model import METEROIAModel


def test_test_forward_has_no_weak_target_argument() -> None:
    parameters = inspect.signature(METEROIAModel.forward).parameters
    assert "meter_grounding" not in parameters
    assert "weak_target" not in parameters
    assert "bdd100k_geometry" not in parameters
