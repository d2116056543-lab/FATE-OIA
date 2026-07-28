from pathlib import Path

from fate_oia.engine.supervise_acpr_meter_oia_foreground import FALLBACK_LADDER


def test_foreground_supervisor_script_exists_and_forbids_hidden_launch() -> None:
    path = Path("scripts/FATE_OIA_acpr_meter_oia_v1_foreground.ps1")
    text = path.read_text(encoding="utf-8")
    assert "Start-Process" not in text
    assert "fate_oia.engine.supervise_acpr_meter_oia_foreground" in text
    assert "fate_oia.engine.train_acpr_meter_oia" not in text


def test_supervisor_validates_ready_payload_and_forbids_full_mock_dino() -> None:
    source = Path(
        "fate_oia/engine/supervise_acpr_meter_oia_foreground.py"
    ).read_text(encoding="utf-8")
    script = Path(
        "scripts/FATE_OIA_acpr_meter_oia_v1_foreground.ps1"
    ).read_text(encoding="utf-8")
    assert "validate_training_readiness" in source
    assert "use_mock_dino=False" in source
    assert "METER pilot/full supervisor requires real DINO" in script


def test_supervisor_starts_with_the_real_dino_profile_winner() -> None:
    assert FALLBACK_LADDER[0] == (6, 5)
    assert FALLBACK_LADDER == ((6, 5), (4, 8), (3, 11), (2, 16))
