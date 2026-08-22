import torch

from fate_oia.models.tida_flow_transition_bank import TIDAFlowTransitionBank


def _linear_trajectory(reverse: bool = False):
    time = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    values = time[:, :, None, None].repeat(1, 1, 2, 4)
    regions = time[:, :, None, None].repeat(1, 1, 2, 3)
    if reverse:
        values = values.flip(1)
        regions = regions.flip(1)
    return values, regions, time, torch.ones(1, 4, dtype=torch.bool)


def test_transition_bank_preserves_signed_velocity_under_reversal():
    module = TIDAFlowTransitionBank(dim=4, region_count=3).eval()
    forward = module(*_linear_trajectory(False))
    reverse = module(*_linear_trajectory(True))

    torch.testing.assert_close(forward["velocity"], -reverse["velocity"], atol=1e-6, rtol=0)
    torch.testing.assert_close(forward["region_velocity"], -reverse["region_velocity"], atol=1e-6, rtol=0)
    assert forward["transition_tokens"].shape == (1, 2, 4)
    assert torch.isfinite(forward["transition_tokens"]).all()


def test_repeated_trajectory_has_zero_motion_and_invalid_steps_are_ignored():
    module = TIDAFlowTransitionBank(dim=4, region_count=3).eval()
    values = torch.ones(2, 5, 2, 4)
    regions = torch.ones(2, 5, 2, 3)
    timestamps = torch.arange(5, dtype=torch.float32).repeat(2, 1)
    valid = torch.tensor([[True, True, True, True, True], [True, True, False, False, True]])
    result = module(values, regions, timestamps, valid)

    assert torch.count_nonzero(result["velocity"]) == 0
    assert torch.count_nonzero(result["acceleration"]) == 0
    assert torch.count_nonzero(result["region_velocity"]) == 0
    assert torch.isfinite(result["transition_reliability"]).all()

