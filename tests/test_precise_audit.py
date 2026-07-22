from pathlib import Path

from fate_oia.engine.audit_precise_oia_implementation import REQUIRED, _scan_forbidden


def test_audit_requires_full_precise_source_surface():
    assert len(REQUIRED) >= 30
    assert "fate_oia/models/precise_oia_model.py" in REQUIRED


def test_forbidden_scan_covers_config_and_script_without_self_matching_audit_rules():
    root = Path(__file__).resolve().parents[1]
    forbidden, incomplete = _scan_forbidden(root)
    assert all("audit_precise_oia_implementation.py" not in paths for paths in (*forbidden.values(), *incomplete.values()))
    assert "Start-Process" not in forbidden
