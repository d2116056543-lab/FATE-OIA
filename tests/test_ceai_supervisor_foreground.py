from pathlib import Path

from fate_oia.engine.audit_ceai_oia_implementation import validate_foreground_files


def test_supervisor_rejects_forbidden_background_keywords():
    errors = validate_foreground_files([
        Path("scripts/FATE_OIA_ceai_oia_v1_foreground.ps1"),
        Path("fate_oia/engine/supervise_ceai_oia_foreground.py"),
    ])
    assert errors == []
