from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ICDORPhase:
    name: str
    route_mode: str
    latent_enabled: bool
    enable_factor_losses: bool
    enable_posterior_ranking: bool
    enable_pareto: bool
    freeze_factor_branch: bool
    pu_enabled: bool = True


def get_icdor_phase(epoch: int, certificate_ready: bool, edge_admission_ready: bool) -> ICDORPhase:
    """The only canonical epoch-to-behavior mapping for IC-DOR."""
    if type(epoch) is not int or epoch not in range(12):
        raise ValueError("IC-DOR canonical schedule requires epoch 0..11")
    if epoch <= 4:
        return ICDORPhase("discovery_shadow", "shadow", True, True, False, False, False)
    if epoch <= 6:
        return ICDORPhase("dual_reason_shadow_action", "shadow", True, True, False, False, False)
    if epoch <= 8:
        return ICDORPhase("safe_action_routing", "admitted" if edge_admission_ready else "shadow", True, True, False, True, False)
    if epoch <= 10:
        return ICDORPhase("joint_ranking", "admitted" if edge_admission_ready else "shadow", True, True, True, True, False)
    return ICDORPhase("consolidation", "admitted" if edge_admission_ready else "shadow", True, True, True, True, False)


def get_icdor_pilot_phase(pilot_epoch: int, certificate_ready: bool, edge_admission_ready: bool) -> ICDORPhase:
    """Four-epoch mechanism pilot; it cannot bypass canonical certificate/admission gates."""
    if type(pilot_epoch) is not int or pilot_epoch not in range(4):
        raise ValueError("IC-DOR pilot schedule requires epoch 0..3")
    if pilot_epoch in {0, 1}:
        return ICDORPhase("pilot_discovery_shadow", "shadow", True, True, False, False, False)
    if pilot_epoch == 2:
        return ICDORPhase("pilot_dual_reason_shadow", "shadow", True, True, False, False, False)
    return ICDORPhase("pilot_safe_route", "admitted" if edge_admission_ready else "shadow", True, True, False, True, False)
