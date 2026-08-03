import inspect

import torch

from fate_oia.models.save_utility_bridge import select_matched_control_patches


def test_matched_control_is_disjoint_equal_sized_and_nonrandom() -> None:
    grid_hw = (8, 12)
    detail_field = torch.zeros(1, grid_hw[0] * grid_hw[1], 4)
    selected = torch.tensor([2 * 12 + 5, 3 * 12 + 6])
    control_expected = torch.tensor([2 * 12 + 4, 3 * 12 + 7])
    detail_field[:, selected, 0] = 1.0
    detail_field[:, control_expected, 0] = 1.0
    detail_field[:, :, 1] = 0.05

    predicate_map = torch.zeros(1, 2, grid_hw[0] * grid_hw[1])
    predicate_map[:, 0, selected] = 1.0
    predicate_map[:, 1, control_expected] = 1.0
    action_contribution = torch.zeros(1, 1, grid_hw[0] * grid_hw[1])
    action_contribution[:, 0, selected] = 1.0
    action_contribution[:, 0, control_expected] = 0.02

    control, metadata = select_matched_control_patches(
        detail_field[0],
        selected,
        grid_hw=grid_hw,
        predicate_map=predicate_map[0],
        action_contribution=action_contribution[0, 0],
    )

    assert control.numel() == selected.numel()
    assert not bool(torch.isin(control, selected).any())
    assert torch.equal(control, control_expected)
    assert metadata["selected_count"] == metadata["control_count"]
    assert metadata["overlap_count"] == 0
    assert metadata["same_sector"] is True
    assert metadata["feature_norm_relative_difference"] < 0.20
    assert metadata["texture_variance_relative_difference"] < 0.25
    assert metadata["predicate_overlap"] < 0.25
    assert metadata["control_action_contribution"] < metadata["selected_action_contribution"]
    assert metadata["selection_method"] == "deterministic_matched_control"


def test_matched_control_has_no_random_fallback_and_rejects_unmatched_pool() -> None:
    source = inspect.getsource(select_matched_control_patches)
    assert "rand" not in source.lower()

    selected = torch.tensor([5, 6])
    valid = torch.zeros(32, dtype=torch.bool)
    valid[selected] = True
    try:
        select_matched_control_patches(
            torch.ones(32, 4),
            selected,
            grid_hw=(4, 8),
            valid_mask=valid,
        )
    except ValueError as error:
        assert "insufficient exact matched controls" in str(error)
    else:
        raise AssertionError("unmatched or self-selected control must be rejected")


def test_matched_control_requires_equal_count_in_each_selected_sector() -> None:
    grid_hw = (6, 6)
    selected = torch.tensor([0, 12])  # depth sectors 0 and 1
    valid = torch.zeros(grid_hw[0] * grid_hw[1], dtype=torch.bool)
    valid[selected] = True
    valid[torch.tensor([1, 6])] = True  # two controls, but both are in sector 0

    try:
        select_matched_control_patches(
            torch.ones(grid_hw[0] * grid_hw[1], 4),
            selected,
            grid_hw=grid_hw,
            valid_mask=valid,
        )
    except ValueError as error:
        assert "insufficient exact matched controls" in str(error)
    else:
        raise AssertionError("control sector counts must exactly match selected sector counts")
