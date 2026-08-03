from pathlib import Path

from fate_oia.engine.audit_save_oia import FORBIDDEN_FORMAL_PATTERNS, REQUIRED_SAVE_FILES, _static_checks


def test_save_audit_declares_concrete_forbidden_paths():
    assert "ACPRPairMemory" in FORBIDDEN_FORMAL_PATTERNS
    assert "cached_logits" in FORBIDDEN_FORMAL_PATTERNS


def test_audit_requires_the_full_save_protocol_surface():
    required = set(REQUIRED_SAVE_FILES)
    assert "fate_oia/engine/evaluate_save_oia_pilot.py" in required
    assert "fate_oia/utils/save_artifacts.py" in required
    assert "scripts/FATE_OIA_save_oia_v1_foreground.ps1" in required
    assert _static_checks(Path.cwd())["required_source_and_protocol_files"] is True
