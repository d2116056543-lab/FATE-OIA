from pathlib import Path


def test_pmt_supervisor_foreground_script_has_no_background_processes():
    text = Path("scripts/FATE_OIA_acpr_pmt_s_v1_foreground.ps1").read_text(encoding="utf-8")
    forbidden = ["Start-Process", "Start-Job", "nohup", "scheduled task"]
    assert not any(x in text for x in forbidden)
    assert "supervise_acpr_pmt_s_foreground" in text
