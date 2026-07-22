from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_foreground_supervisor_script_has_no_background_mechanism():
    script = (ROOT / "scripts" / "FATE_OIA_precise_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    for forbidden in ("Start-Process", "Start-Job", "nohup", "Register-ScheduledTask"):
        assert forbidden not in script
    assert "supervise_precise_oia_foreground" in script
