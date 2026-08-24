import torch

from fate_oia.utils.tida_traffic_calibration import (
    apply_action_traffic_calibration,
    apply_action_traffic_utility,
    fit_action_traffic_calibration,
    fit_action_traffic_calibration_oof,
    fit_action_traffic_utility_oof,
)


def test_action_specific_traffic_calibration_can_choose_distinct_scales():
    target = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    semantic = torch.zeros(4, 2)
    delta = torch.tensor([[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]])
    result = fit_action_traffic_calibration(semantic, delta, target, candidates=(-1.0, 0.0, 1.0))
    torch.testing.assert_close(result["scales"], torch.tensor([1.0, -1.0]))
    assert torch.all((result["thresholds"] >= 0) & (result["thresholds"] <= 1))


def test_oof_trajectory_calibration_selects_useful_scale_and_is_reproducible():
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]]).repeat(8, 1)
    semantic = torch.zeros_like(target)
    delta = (2.0 * target - 1.0) * 0.1
    result = fit_action_traffic_calibration_oof(
        semantic, delta, target, candidates=(0.0, 1.0, 4.0), folds=4
    )
    assert result["scales"].item() == 1.0
    assert result["oof_gain_by_action"].item() > 0.3
    deployed = apply_action_traffic_calibration(semantic, delta, result["scales"])
    assert torch.equal(deployed > 0, target.bool())


def test_oof_trajectory_calibration_falls_back_to_zero_when_delta_is_harmful():
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]]).repeat(8, 1)
    semantic = (2.0 * target - 1.0) * 0.2
    harmful = torch.zeros_like(target)
    result = fit_action_traffic_calibration_oof(
        semantic, harmful, target, candidates=(0.0, 1.0, 2.0), folds=4
    )
    assert result["scales"].item() == 0.0
    assert result["oof_gain_by_action"].item() == 0.0


def test_oof_trajectory_calibration_can_correct_action_specific_orientation():
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]]).repeat(8, 1)
    semantic = torch.zeros_like(target)
    reversed_delta = -(2.0 * target - 1.0) * 0.1
    result = fit_action_traffic_calibration_oof(
        semantic, reversed_delta, target, candidates=(-1.0, 0.0, 1.0), folds=4
    )
    assert result["scales"].item() == -1.0
    assert result["oof_gain_by_action"].item() > 0.3


def test_oof_utility_calibration_filters_harmful_trajectory_candidates():
    target = torch.tensor([[1.0], [0.0]]).repeat(40, 1)
    sign = 2.0 * target - 1.0
    helpful = torch.arange(target.shape[0]) % 4 != 0
    candidate = torch.where(helpful[:, None], sign, -sign) * 0.2
    utility = torch.where(helpful[:, None], torch.tensor(0.9), torch.tensor(0.1))
    semantic = torch.zeros_like(target)
    result = fit_action_traffic_utility_oof(
        semantic, candidate, utility, target,
        cutoffs=(0.0, 0.5), scales=(0.0, 1.0), folds=4,
    )
    assert result["cutoffs"].item() == 0.5
    assert result["scales"].item() == 1.0
    deployed = apply_action_traffic_utility(
        semantic, candidate, utility, result["scales"], result["cutoffs"]
    )
    assert result["oof_gain_by_action"].item() > 0.3
    assert deployed[helpful].abs().mean() > deployed[~helpful].abs().mean()


def test_oof_utility_calibration_has_strict_zero_fallback():
    target = torch.tensor([[1.0], [0.0]]).repeat(20, 1)
    semantic = (2.0 * target - 1.0) * 0.2
    result = fit_action_traffic_utility_oof(
        semantic, torch.zeros_like(target), torch.full_like(target, 0.5), target,
        cutoffs=(0.0, 0.5), scales=(0.0, 1.0), folds=4,
    )
    assert result["scales"].item() == 0.0
    assert result["oof_gain_by_action"].item() == 0.0
