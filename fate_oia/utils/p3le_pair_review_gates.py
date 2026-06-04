from __future__ import annotations

from pathlib import Path


FORBIDDEN_BACKGROUND_TOKENS = ("Start-Process", "Start-Job", "nohup", "-WindowStyle Hidden")


def assert_foreground_script(path: str | Path) -> None:
    text = Path(path).read_text(encoding="utf-8")
    lowered = text.lower()
    for token in FORBIDDEN_BACKGROUND_TOKENS:
        if token.lower() in lowered:
            raise AssertionError(f"foreground script contains forbidden background token: {token}")


def assert_no_extra_status_markdown(worktree: str | Path) -> None:
    allowed = {"README.md"}
    for path in Path(worktree).rglob("*.md"):
        if ".git" in path.parts:
            continue
        if path.name in allowed:
            continue
        if "docs" in path.parts:
            continue
        raise AssertionError(f"unexpected durable markdown in worktree: {path}")
