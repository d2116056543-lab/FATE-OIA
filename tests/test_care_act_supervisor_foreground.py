from __future__ import annotations

from pathlib import Path


def test_care_act_supervisor_does_not_use_detached_process_tokens():
    src = Path("fate_oia/engine/supervise_care_act_oia_foreground.py").read_text(encoding="utf-8")
    forbidden = ["Start-Process", "Start-Job", "Win32_Process", "Invoke-WmiMethod", "nohup", "hidden cmd"]
    assert not any(x in src for x in forbidden)
    assert "subprocess.Popen" in src
    assert ".wait()" in src or "proc.wait" in src
