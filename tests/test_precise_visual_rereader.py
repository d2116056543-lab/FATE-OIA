import torch

from fate_oia.models.precise_visual_rereader import PRECISEVisualRereader


def _evidence(batch=2):
    coordinates = torch.rand(batch, 10, 8, 2)
    return {"explicit_tokens": torch.randn(batch, 10, 384), "part_coordinates": coordinates}


def test_rereader_uses_evidence_coordinates_without_a_second_backbone_call():
    model = PRECISEVisualRereader()
    action = torch.randn(2, 4, 384)
    reason = torch.randn(2, 21, 384)
    output = model(action, reason, _evidence(), torch.randn(2, 3, 3600, 384), torch.randn(2, 3, 3600, 384))
    assert output["reference_points"].shape == (2, 25, 3, 4, 2)
    assert output["action_reread_delta"].shape == (2, 4, 384)
    assert output["reason_reread_delta"].shape == (2, 21, 384)
    assert torch.isfinite(output["reference_point_variance"]).all()
