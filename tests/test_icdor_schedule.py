from __future__ import annotations

import pytest

from fate_oia.engine.mosaic_icdor_schedule import get_icdor_phase, get_icdor_pilot_phase


def test_icdor_canonical_schedule_grants_shadow_learning_without_certificate() -> None:
    foundation = get_icdor_phase(0, certificate_ready=False, edge_admission_ready=False)
    discovery = get_icdor_phase(4, certificate_ready=False, edge_admission_ready=False)
    assert foundation.name == "discovery_shadow"
    assert foundation.route_mode == "shadow"
    assert foundation.latent_enabled is True
    assert discovery.name == "discovery_shadow"

    dual = get_icdor_phase(5, certificate_ready=False, edge_admission_ready=False)
    assert dual.route_mode == "shadow"
    safe_shadow = get_icdor_phase(7, certificate_ready=False, edge_admission_ready=False)
    assert safe_shadow.route_mode == "shadow"
    safe = get_icdor_phase(7, certificate_ready=False, edge_admission_ready=True)
    assert safe.route_mode == "admitted"
    assert safe.enable_pareto is True


def test_icdor_pilot_schedule_trains_shadow_before_optional_safe_route() -> None:
    assert get_icdor_pilot_phase(0, certificate_ready=False, edge_admission_ready=False).name == "pilot_discovery_shadow"
    assert get_icdor_pilot_phase(1, certificate_ready=False, edge_admission_ready=False).name == "pilot_discovery_shadow"
    assert get_icdor_pilot_phase(2, certificate_ready=False, edge_admission_ready=False).name == "pilot_dual_reason_shadow"
    assert get_icdor_pilot_phase(3, certificate_ready=False, edge_admission_ready=False).route_mode == "shadow"
    safe = get_icdor_pilot_phase(3, certificate_ready=False, edge_admission_ready=True)
    assert safe.route_mode == "admitted"
    assert safe.enable_pareto is True
