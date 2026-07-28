from pathlib import Path


def test_foreground_supervisor_script_exists_and_forbids_hidden_launch() -> None:
    path = Path("scripts/FATE_OIA_acpr_meter_oia_v1_foreground.ps1")
    text = path.read_text(encoding="utf-8")
    assert "Start-Process" not in text
    assert "fate_oia.engine.train_acpr_meter_oia" in text
