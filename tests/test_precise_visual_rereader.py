import torch

from fate_oia.models.precise_visual_rereader import PRECISEVisualRereader


def _evidence(batch=2):
    coordinates = torch.rand(batch, 10, 8, 2)
    part_valid = torch.ones(10, 8, dtype=torch.bool)
    part_valid[:5, 4:] = False
    return {"explicit_tokens": torch.randn(batch, 10, 384), "part_coordinates": coordinates, "part_valid": part_valid, "reliability": torch.rand(batch, 10)}


def test_rereader_uses_evidence_coordinates_without_a_second_backbone_call():
    model = PRECISEVisualRereader()
    action = torch.randn(2, 4, 384)
    reason = torch.randn(2, 21, 384)
    output = model(action, reason, _evidence(), torch.randn(2, 3, 3600, 384), torch.randn(2, 3, 3600, 384), torch.randn(2, 4), torch.randn(2, 21))
    assert output["reference_points"].shape == (2, 25, 3, 4, 2)
    assert output["action_reread_delta"].shape == (2, 4, 384)
    assert output["reason_reread_delta"].shape == (2, 21, 384)
    assert torch.isfinite(output["reference_point_variance"]).all()
    assert output["evidence_demand_state"].shape == (2, 25, 6)
    assert (output["sampling_weights"] > 0).sum(-1).max().item() <= 2


def test_rereader_ignores_padded_parts_and_reports_raw_out_of_bounds_points():
    model = PRECISEVisualRereader()
    evidence = _evidence(1)
    evidence["part_coordinates"][:] = 0.99
    evidence["part_coordinates"][:, :5, 4:] = 0.0
    with torch.no_grad():
        model.error_mlp[-1].bias.fill_(10.0)
    output = model(
        torch.randn(1, 4, 384), torch.randn(1, 21, 384), evidence,
        torch.randn(1, 3, 3600, 384), torch.randn(1, 3, 3600, 384),
        torch.randn(1, 4), torch.randn(1, 21),
    )
    assert output["raw_reference_points"].amax() > 1.0
    assert output["out_of_bounds_rate"] > 0
    assert output["reference_points"].amin() >= 0 and output["reference_points"].amax() <= 1
