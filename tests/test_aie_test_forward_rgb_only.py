import inspect

from fate_oia.models.aie_oia_model import AIEOIAModel


def test_formal_forward_accepts_no_labels_or_structured_records():
    parameters = inspect.signature(AIEOIAModel.forward).parameters
    assert set(parameters) == {"self", "images", "action_scale", "reason_scale"}

