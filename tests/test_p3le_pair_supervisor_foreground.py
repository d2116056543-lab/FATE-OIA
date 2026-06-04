from pathlib import Path

from fate_oia.utils.p3le_pair_review_gates import assert_foreground_script


def test_supervisor_and_powershell_script_forbid_background_launching():
    assert_foreground_script(Path("fate_oia/engine/supervise_p3le_pair_oia_foreground.py"))
    assert_foreground_script(Path("scripts/FATE_OIA_p3le_pair_oia_v1_foreground.ps1"))
    script = Path("scripts/FATE_OIA_p3le_pair_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    assert "Start-Process" not in script
    assert "Start-Job" not in script
