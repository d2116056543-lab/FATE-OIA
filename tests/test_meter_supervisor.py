from pathlib import Path


def test_foreground_supervisor_script_exists_and_forbids_hidden_launch() -> None:
    path = Path("scripts/FATE_OIA_acpr_meter_oia_v1_foreground.ps1")
    text = path.read_text(encoding="utf-8")
    assert "Start-Process" not in text
    assert "fate_oia.engine.train_acpr_meter_oia" in text


def test_supervisor_validates_ready_payload_and_forbids_full_mock_dino() -> None:
    source = Path(
        "fate_oia/engine/supervise_acpr_meter_oia_foreground.py"
    ).read_text(encoding="utf-8")
    script = Path(
        "scripts/FATE_OIA_acpr_meter_oia_v1_foreground.ps1"
    ).read_text(encoding="utf-8")
    assert "validate_training_readiness" in source
    assert "use_mock_dino=False" in source
    assert "Full training cannot use mock DINO" in script
