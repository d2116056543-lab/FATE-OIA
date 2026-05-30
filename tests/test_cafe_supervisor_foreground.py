from __future__ import annotations

from pathlib import Path


def test_supervisor_rejects_background():
    root = Path(__file__).resolve().parents[1]
    text = (root / "fate_oia" / "engine" / "supervise_cafe_oia_foreground.py").read_text(encoding="utf-8")
    script = (root / "scripts" / "FATE_OIA_clean_cafe_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    combined = text + "\n" + script
    for forbidden in ["Start-Process", "Start-Job", "Win32_Process.Create", "nohup"]:
        assert forbidden not in combined
    assert "subprocess.Popen" in text
    assert "stdout=subprocess.PIPE" in text

