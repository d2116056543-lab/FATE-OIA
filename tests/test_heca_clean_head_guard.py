from fate_oia.utils.heca_clean_head import worktree_admission_failures


def test_clean_head_guard_rejects_tracked_changes_and_untracked_runtime_source() -> None:
    status = " M fate_oia/models/meter_oia_model.py\n?? fate_oia/engine/rogue_runtime.py\n?? configs/rogue.yaml\n?? scripts/rogue.ps1\n?? docs/review_notes.md\n"

    failures = worktree_admission_failures(status)

    assert "tracked_changes_present" in failures
    assert "untracked_runtime_source:fate_oia/engine/rogue_runtime.py" in failures
    assert "untracked_runtime_source:configs/rogue.yaml" in failures
    assert "untracked_runtime_source:scripts/rogue.ps1" in failures
    assert all("docs/review_notes.md" not in failure for failure in failures)


def test_clean_head_guard_allows_only_untracked_non_runtime_records() -> None:
    status = "?? docs/superpowers/supervision/review.md\n?? .heca_root_cause_fix.patch\n"

    assert worktree_admission_failures(status) == []
