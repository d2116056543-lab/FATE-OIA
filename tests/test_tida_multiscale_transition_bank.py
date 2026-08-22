import torch

from fate_oia.models.tida_flow_transition_bank import TIDAFlowTransitionBank


def _inputs(valid: bool = True):
    timestamps = torch.arange(5, dtype=torch.float32).repeat(2, 1)
    trajectory = timestamps[:, :, None, None].repeat(1, 1, 3, 8)
    regions = timestamps[:, :, None, None].repeat(1, 1, 3, 5)
    mask = torch.full((2, 5), valid, dtype=torch.bool)
    return trajectory, regions, timestamps, mask


def test_transition_bank_returns_four_typed_scales():
    output = TIDAFlowTransitionBank(dim=8, region_count=5)(*_inputs())

    assert output["transition_tokens_by_scale"].shape == (2, 3, 4, 8)
    assert output["transition_scale_names"] == (
        "velocity",
        "acceleration",
        "region_velocity",
        "persistence",
    )
    torch.testing.assert_close(
        output["transition_tokens_by_scale"],
        output["transition_tokens"][:, :, None].expand(-1, -1, 4, -1),
    )


def test_transition_token_is_always_the_mean_of_typed_scales():
    bank = TIDAFlowTransitionBank(dim=8, region_count=5)
    with torch.no_grad():
        bank.scale_residuals[0][-1].bias.fill_(1.0)
    output = bank(*_inputs(valid=True))
    torch.testing.assert_close(output["transition_tokens"], output["transition_tokens_by_scale"].mean(2))
    assert not torch.equal(output["transition_tokens"], output["legacy_transition_tokens"])


def test_motion_salience_is_finite_non_saturated_and_zero_without_history():
    module = TIDAFlowTransitionBank(dim=8, region_count=5)
    real = module(*_inputs())
    empty = module(*_inputs(valid=False))

    assert torch.isfinite(real["motion_salience"]).all()
    assert real["motion_salience"].shape == (2, 3)
    assert torch.all(real["motion_salience"] > 0)
    assert torch.isfinite(real["transition_consistency"]).all()
    assert torch.count_nonzero(empty["motion_salience"]) == 0
    assert torch.count_nonzero(empty["transition_consistency"]) == 0
