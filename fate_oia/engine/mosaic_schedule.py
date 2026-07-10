from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MOSAICPhaseControls:
    epoch: int
    phase: str
    state_residual_scale: float
    action_state_gate_cap: float
    reason_state_contribution_cap: float
    learned_propensity: bool
    posterior_enabled: bool
    synthetic_missing_positive: bool
    posterior_rank_weight_scale: float
    action_anchor_enabled: bool
    freeze_factor_prototypes: bool
    freeze_propensity_groups: bool
    representation_lr_scale: float
    calibration_only: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def mosaic_phase_controls(epoch: int, *, total_epochs: int = 15) -> MOSAICPhaseControls:
    if type(epoch) is not int or epoch < 0 or epoch >= total_epochs:
        raise ValueError(f"epoch must be an integer in [0,{total_epochs - 1}]")
    if total_epochs != 15:
        raise ValueError("formal MOSAIC-AD schedule is fixed to 15 epochs")

    if epoch <= 2:
        return MOSAICPhaseControls(
            epoch, "A_visual_foundation", 0.0, 0.0, 0.0, False, False, False,
            0.0, False, False, False, 1.0, False,
        )
    if epoch <= 5:
        step = (epoch - 3) / 2.0
        return MOSAICPhaseControls(
            epoch, "B_state_composition", 0.10 * step, 0.15 * step, 0.10 * step,
            False, False, False, 0.0, True, False, False, 1.0, False,
        )
    if epoch <= 8:
        rank_scale = (epoch - 5) / 3.0
        return MOSAICPhaseControls(
            epoch, "C_selective_observation", 0.10, 0.15, 0.10,
            True, True, True, rank_scale, True, False, False, 1.0, False,
        )
    if epoch <= 11:
        step = (epoch - 9) / 2.0
        return MOSAICPhaseControls(
            epoch, "D_joint_ranking", 0.10, 0.15 + 0.10 * step, 0.10 + 0.10 * step,
            True, True, True, 1.0, True, False, False, 1.0, False,
        )
    if epoch == 12:
        return MOSAICPhaseControls(
            epoch, "E_representation_consolidation", 0.10, 0.25, 0.20,
            True, True, True, 1.0, True, True, True, 0.20, False,
        )
    return MOSAICPhaseControls(
        epoch, "F_calibration_only", 0.10, 0.25, 0.20,
        False, False, False, 0.0, False, True, True, 0.0, True,
    )
