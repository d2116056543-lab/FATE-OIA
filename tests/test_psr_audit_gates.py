from __future__ import annotations

from pathlib import Path

import pytest

from fate_oia.utils.psr_review_gates import assert_foreground_only


def test_foreground_gate_blocks_background_tokens(tmp_path):
    ok = tmp_path / "ok.ps1"
    ok.write_text("& $Python -m fate_oia.engine.supervise_psr_oia_goal", encoding="utf-8")
    assert_foreground_only([ok])
    bad = tmp_path / "bad.ps1"
    bad.write_text("Start-Process python", encoding="utf-8")
    with pytest.raises(ValueError):
        assert_foreground_only([bad])
