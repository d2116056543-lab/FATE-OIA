from __future__ import annotations

import pytest

from fate_oia.engine.mosaic_schedule import mosaic_phase_controls


def test_formal_schedule_has_exact_six_phases_and_boundaries() -> None:
    phases = [mosaic_phase_controls(epoch).phase for epoch in range(15)]
    assert phases == (
        ["A_visual_foundation"] * 3
        + ["B_state_composition"] * 3
        + ["C_selective_observation"] * 3
        + ["D_joint_ranking"] * 3
        + ["E_representation_consolidation"]
        + ["F_calibration_only"] * 2
    )


def test_phase_a_has_hard_zero_state_and_no_selective_observation() -> None:
    for epoch in range(3):
        phase = mosaic_phase_controls(epoch)
        assert phase.state_residual_scale == 0
        assert phase.action_state_gate_cap == 0
        assert not phase.learned_propensity
        assert not phase.posterior_enabled
        assert not phase.synthetic_missing_positive
        assert phase.posterior_rank_weight_scale == 0


def test_phase_b_ramps_state_but_keeps_propensity_fixed() -> None:
    start = mosaic_phase_controls(3)
    end = mosaic_phase_controls(5)
    assert start.state_residual_scale == 0
    assert start.action_state_gate_cap == 0
    assert end.state_residual_scale == pytest.approx(0.10)
    assert end.action_state_gate_cap == pytest.approx(0.15)
    assert end.action_anchor_enabled
    assert not end.learned_propensity


def test_phase_c_enables_exact_posterior_and_warms_ranking() -> None:
    scales = [mosaic_phase_controls(epoch).posterior_rank_weight_scale for epoch in range(6, 9)]
    assert scales == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert all(mosaic_phase_controls(epoch).synthetic_missing_positive for epoch in range(6, 9))


def test_phase_d_caps_state_contributions_and_phase_e_consolidates() -> None:
    end = mosaic_phase_controls(11)
    assert end.action_state_gate_cap == pytest.approx(0.25)
    assert end.reason_state_contribution_cap == pytest.approx(0.20)
    consolidation = mosaic_phase_controls(12)
    assert consolidation.freeze_factor_prototypes
    assert consolidation.freeze_propensity_groups
    assert consolidation.representation_lr_scale == pytest.approx(0.20)


def test_phase_f_is_calibration_only() -> None:
    for epoch in (13, 14):
        phase = mosaic_phase_controls(epoch)
        assert phase.calibration_only
        assert phase.representation_lr_scale == 0
        assert not phase.action_anchor_enabled
        assert not phase.posterior_enabled


@pytest.mark.parametrize("epoch", [-1, 15])
def test_out_of_range_epoch_fails_closed(epoch: int) -> None:
    with pytest.raises(ValueError):
        mosaic_phase_controls(epoch)
