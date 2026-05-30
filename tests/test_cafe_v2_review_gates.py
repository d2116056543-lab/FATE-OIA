from __future__ import annotations

from pathlib import Path

from fate_oia.utils.cafe_review_gates import FORBIDDEN_FOREGROUND_TOKENS, FORBIDDEN_PLACEHOLDERS, scan_forbidden_tokens


def test_foreground_script_has_no_detach_tokens() -> None:
    hits = scan_forbidden_tokens([Path("scripts/FATE_OIA_clean_cafe_oia_v2_foreground.ps1")], FORBIDDEN_FOREGROUND_TOKENS)
    assert hits == {}


def test_v2_files_have_no_placeholder_tokens() -> None:
    paths = [
        Path("fate_oia/engine/train_cafe_oia.py"),
        Path("fate_oia/models/cafe_oia_model.py"),
        Path("fate_oia/engine/calibrate_cafe_oia.py"),
    ]
    hits = scan_forbidden_tokens(paths, FORBIDDEN_PLACEHOLDERS)
    assert hits == {}
