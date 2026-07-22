from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_foreground_supervisor_script_has_no_background_mechanism():
    script = (ROOT / "scripts" / "FATE_OIA_precise_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    for forbidden in ("Start-Process", "Start-Job", "nohup", "Register-ScheduledTask"):
        assert forbidden not in script
    assert "supervise_precise_oia_foreground" in script


def test_supervisor_profiles_real_complete_path_before_preflight_audit():
    source = (ROOT / "fate_oia" / "engine" / "supervise_precise_oia_foreground.py").read_text(encoding="utf-8")
    assert "fate_oia.engine.profile_precise_oia" in source
    assert source.index("_run(profile)") < source.index("_run(audit)")
    assert "review_is_current(" in source


def test_review_gate_verification_checks_status_source_tree_and_pilot_checks():
    source = (ROOT / "fate_oia" / "engine" / "supervise_precise_oia_foreground.py").read_text(encoding="utf-8")
    for required in ("expected_status", "source_tree_sha256", "functional_checks", "pilot_checks"):
        assert required in source
    assert 'expected_status="FULL_TRAIN_READY"' in source
