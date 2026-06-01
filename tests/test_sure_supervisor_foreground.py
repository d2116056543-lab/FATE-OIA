from __future__ import annotations

from pathlib import Path

from fate_oia.utils.sure_review_gates import FORBIDDEN_SUPERVISOR_PATTERNS


def test_supervisor_has_no_disallowed_launcher_patterns() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ["fate_oia/engine/supervise_sure_oia_foreground.py", "scripts/FATE_OIA_sure_oia_v2_foreground.ps1"]:
        text = (root / rel).read_text(encoding="utf-8")
        for pattern in FORBIDDEN_SUPERVISOR_PATTERNS:
            assert pattern not in text
