from pathlib import Path

from fate_oia.engine.supervise_acpr_meter_oia_foreground import (
    FALLBACK_LADDER,
    PILOT_GATES,
)


def test_tesa_foreground_supervisor_forbids_hidden_launch():
    path = Path("scripts/FATE_OIA_acpr_meter_oia_v2_tesa_foreground.ps1")
    text = path.read_text(encoding="utf-8")
    assert "Start-Process" not in text
    assert "Start-Job" not in text
    assert "supervise_acpr_meter_oia_foreground" in text


def test_tesa_supervisor_requires_all_pilot_gates_and_uses_fixed_ladder():
    assert PILOT_GATES == tuple("ABCDEFGH")
    assert FALLBACK_LADDER == ((6, 5), (5, 6), (4, 8), (3, 10), (2, 15))
