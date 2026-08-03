from __future__ import annotations

from pathlib import Path

from fate_oia.utils.save_contracts import (
    SAVE_LOCAL_MIRROR_SUFFIX,
    L08_WRITE_SET,
    SAVE_TARGET_WORKTREE_SUFFIX,
    validate_save_worktree,
)


def test_save_worktree_and_l08_write_set_are_explicit() -> None:
    root = Path(__file__).resolve().parents[1]
    assert SAVE_TARGET_WORKTREE_SUFFIX == "fate_oia_acpr_save_oia_v1_worktree"
    assert root.name in {SAVE_TARGET_WORKTREE_SUFFIX, SAVE_LOCAL_MIRROR_SUFFIX}
    assert validate_save_worktree(root)
    assert L08_WRITE_SET == frozenset(
        {
            "configs/fate_oia_train_360x640_save_oia_v1.yaml",
            "configs/save_factor_schema.yaml",
            "fate_oia/models/save_oia_model.py",
            "fate_oia/utils/save_contracts.py",
            "fate_oia/models/meter_calalign_foundation.py",
            "tests/test_save_source_head_contract.py",
            "tests/test_save_worktree_contract.py",
            "tests/test_save_forbidden_paths.py",
            "tests/test_save_full_calalign_equivalence.py",
            "tests/test_save_uses_calalign_fused_action.py",
            "tests/test_save_one_dino_call.py",
            "tests/test_save_same_forward_branches.py",
            "tests/test_save_test_forward_image_only.py",
        }
    )
