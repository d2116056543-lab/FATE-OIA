from __future__ import annotations

import subprocess
from pathlib import Path

from fate_oia.utils.save_contracts import (
    SAVE_SOURCE_HEAD,
    SAVE_TARGET_BRANCH,
    validate_save_source_head,
)


def test_save_source_head_is_the_approved_calalign_ancestor() -> None:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()

    assert branch == SAVE_TARGET_BRANCH
    assert len(SAVE_SOURCE_HEAD) == 40
    assert validate_save_source_head(root, head=head)
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", SAVE_SOURCE_HEAD, head],
        cwd=root,
        check=False,
    ).returncode == 0
