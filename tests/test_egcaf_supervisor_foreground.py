from pathlib import Path


def test_supervisor_and_ps1_are_foreground_only():
    paths = ["scripts/FATE_OIA_egcaf_oia_v1_foreground.ps1", "fate_oia/engine/supervise_egcaf_oia_foreground.py"]
    banned = ["Start-Process", "Start-Job", "nohup", "disown", "WindowStyle Hidden"]
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for b in banned:
            assert b not in text
