from __future__ import annotations

from pathlib import Path

from fate_oia.utils.psr_review_gates import FORBIDDEN_FOREGROUND_TOKENS


def test_psr_supervisor_and_script_are_foreground_only():
    paths = [
        Path("scripts/FATE_OIA_psr_oia_v2_goal.ps1"),
        Path("fate_oia/engine/supervise_psr_oia_goal.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_FOREGROUND_TOKENS:
            assert token not in text
    assert "require_review_pass" in paths[1].read_text(encoding="utf-8")
