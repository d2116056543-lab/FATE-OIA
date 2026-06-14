from pathlib import Path


def test_acpr_supervisor_script_foreground_only():
    text = Path("scripts/FATE_OIA_acpr_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    assert "Start-Process" not in text
    assert "Start-Job" not in text
    assert "nohup" not in text
