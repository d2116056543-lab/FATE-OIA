from __future__ import annotations

from pathlib import Path


def test_vista_supervisor_script_avoids_background_primitives():
    text = Path("scripts/FATE_OIA_acpr_vista_v1_foreground.ps1").read_text(encoding="utf-8")
    forbidden = ["Start-Process", "Start-Job", "nohup", "scheduled task", "hidden cmd"]
    assert not any(x in text for x in forbidden)

