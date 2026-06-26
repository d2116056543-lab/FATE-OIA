from pathlib import Path


def test_gem_supervisor_script_is_foreground_only():
    text = Path("scripts/FATE_OIA_acpr_gem_v1_foreground.ps1").read_text(encoding="utf-8")
    forbidden = ["Start-Process", "Start-Job", "nohup", "scheduled task", "hidden cmd"]
    for pat in forbidden:
        assert pat not in text
    assert "supervise_acpr_gem_foreground" in text
