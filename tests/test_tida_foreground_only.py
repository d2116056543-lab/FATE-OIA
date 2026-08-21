from pathlib import Path


def test_supervisor_script_is_synchronous_and_visible():
    source = Path("scripts/FATE_OIA_tida_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    for forbidden in ("Start-Process", "Start-Job", "nohup", "Hidden", "scheduled task"):
        assert forbidden.lower() not in source.lower()
    assert "supervise_tida_oia_foreground" in source
