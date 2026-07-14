from __future__ import annotations

import pytest

from fate_oia.engine.mosaic_icdor_schedule import get_icdor_phase, get_icdor_pilot_phase


def test_icdor_canonical_schedule_is_centralized_and_certificate_gated() -> None:
    foundation = get_icdor_phase(0, certificate_ready=False, edge_admission_ready=False)
    certification = get_icdor_phase(4, certificate_ready=False, edge_admission_ready=False)
    assert foundation.name == "visual_foundation"
    assert foundation.route_mode == "off"
    assert certification.name == "factor_certification"

    with pytest.raises(RuntimeError, match="certificate"):
        get_icdor_phase(5, certificate_ready=False, edge_admission_ready=False)
    dual = get_icdor_phase(5, certificate_ready=True, edge_admission_ready=False)
    assert dual.route_mode == "shadow"
    with pytest.raises(RuntimeError, match="edge admission"):
        get_icdor_phase(7, certificate_ready=True, edge_admission_ready=False)
    safe = get_icdor_phase(7, certificate_ready=True, edge_admission_ready=True)
    assert safe.route_mode == "admitted"
    assert safe.enable_pareto is True


def test_icdor_pilot_schedule_builds_edge_after_epoch2_then_exercises_safe_route() -> None:
    assert get_icdor_pilot_phase(0, certificate_ready=False, edge_admission_ready=False).name == "pilot_foundation"
    assert get_icdor_pilot_phase(1, certificate_ready=False, edge_admission_ready=False).name == "pilot_certificate_diagnostic"
    with pytest.raises(RuntimeError, match="certificate"):
        get_icdor_pilot_phase(2, certificate_ready=False, edge_admission_ready=False)
    assert get_icdor_pilot_phase(2, certificate_ready=True, edge_admission_ready=False).route_mode == "shadow"
    with pytest.raises(RuntimeError, match="edge admission"):
        get_icdor_pilot_phase(3, certificate_ready=True, edge_admission_ready=False)
    safe = get_icdor_pilot_phase(3, certificate_ready=True, edge_admission_ready=True)
    assert safe.route_mode == "admitted"
    assert safe.enable_pareto is True
