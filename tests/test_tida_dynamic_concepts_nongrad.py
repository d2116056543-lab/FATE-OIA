import torch

from fate_oia.explain.tida_dynamic_concepts import translate_dynamic_concepts


def test_dynamic_concepts_are_plain_nongrad_records():
    concepts = translate_dynamic_concepts(
        ["front_vehicle_close", "vehicle_left"],
        torch.tensor([[[0.0, 0.0, 0.2, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0]]], requires_grad=True),
        torch.tensor([[0.8, 0.8]], requires_grad=True),
    )
    assert isinstance(concepts, list) and isinstance(concepts[0], dict)
    assert all(not isinstance(value, torch.Tensor) for value in concepts[0].values())
    assert concepts[0]["front_vehicle_close"] == "front_closing"
    assert concepts[0]["vehicle_left"] == "left_inflow"


def test_dynamic_concepts_can_report_both_motion_directions():
    concepts = translate_dynamic_concepts(
        ["front_vehicle_close"], torch.tensor([[[0.0, 0.0, -0.2, 0.0, 0.0]]]), torch.ones(1, 1)
    )
    assert concepts[0]["front_vehicle_close"] == "front_receding"
