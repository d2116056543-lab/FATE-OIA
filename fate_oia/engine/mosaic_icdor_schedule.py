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


def get_icdor_phase(epoch: int, certificate_ready: bool, edge_admission_ready: bool) -> ICDORPhase:
    """The only canonical epoch-to-behavior mapping for IC-DOR."""
    if type(epoch) is not int or epoch not in range(12):
        raise ValueError("IC-DOR canonical schedule requires epoch 0..11")
    if epoch <= 2:
        return ICDORPhase("visual_foundation", "off", False, True, False, False, False)
    if epoch <= 4:
        return ICDORPhase("factor_certification", "off", False, True, False, False, False)
    if not certificate_ready:
        raise RuntimeError("IC-DOR certificate must be ready after epoch 4 before dual-reason training")
    if epoch <= 6:
        return ICDORPhase("dual_reason_shadow_action", "shadow", True, False, False, False, True)
    if not edge_admission_ready:
        raise RuntimeError("IC-DOR edge admission must be ready after epoch 6 before safe action routing")
    if epoch <= 8:
        return ICDORPhase("safe_action_routing", "admitted", True, False, False, True, True)
    if epoch <= 10:
        return ICDORPhase("joint_ranking", "admitted", True, False, True, True, True)
    return ICDORPhase("consolidation", "admitted", True, False, True, True, True)


def get_icdor_pilot_phase(pilot_epoch: int, certificate_ready: bool, edge_admission_ready: bool) -> ICDORPhase:
    """Four-epoch mechanism pilot; it cannot bypass canonical certificate/admission gates."""
    if type(pilot_epoch) is not int or pilot_epoch not in range(4):
        raise ValueError("IC-DOR pilot schedule requires epoch 0..3")
    if pilot_epoch == 0:
        return ICDORPhase("pilot_foundation", "off", False, True, False, False, False)
    if pilot_epoch == 1:
        # Diagnostic certificate construction is allowed here, but cannot route actions.
        return ICDORPhase("pilot_certificate_diagnostic", "off", False, True, False, False, False)
    if not certificate_ready:
        raise RuntimeError("IC-DOR pilot dual-reason stage requires a certificate")
    if pilot_epoch == 2:
        return ICDORPhase("pilot_dual_reason_shadow", "shadow", True, False, False, False, True)
    if not edge_admission_ready:
        raise RuntimeError("IC-DOR pilot safe-route stage requires edge admission from epoch 2")
    return ICDORPhase("pilot_safe_route", "admitted", True, False, False, True, True)
