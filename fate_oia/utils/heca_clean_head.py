from __future__ import annotations


_RUNTIME_SOURCE_PREFIXES = ("fate_oia/", "configs/", "scripts/")


def worktree_admission_failures(status_porcelain: str) -> list[str]:
    """Reject code that is not represented by the audited commit HEAD."""
    failures: list[str] = []
    for raw_line in status_porcelain.splitlines():
        if not raw_line:
            continue
        if not raw_line.startswith("?? "):
            failures.append("tracked_changes_present")
            continue
        path = raw_line[3:].strip().replace("\\", "/")
        if path.startswith(_RUNTIME_SOURCE_PREFIXES):
            failures.append(f"untracked_runtime_source:{path}")
    return sorted(set(failures))
