import torch

from fate_oia.utils.tida_traffic_calibration import fit_action_traffic_calibration


def test_action_specific_traffic_calibration_can_choose_distinct_scales():
    target = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    semantic = torch.zeros(4, 2)
    delta = torch.tensor([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
    result = fit_action_traffic_calibration(semantic, delta, target, candidates=(-1.0, 0.0, 1.0))
    torch.testing.assert_close(result["scales"], torch.tensor([1.0, -1.0]))
    assert torch.all((result["thresholds"] >= 0) & (result["thresholds"] <= 1))
