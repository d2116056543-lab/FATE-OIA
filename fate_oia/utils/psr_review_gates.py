from __future__ import annotations

from pathlib import Path


FORBIDDEN_FOREGROUND_TOKENS = ["Start-Process", "Start-Job", "Win32_Process", "Invoke-WmiMethod", "nohup", "-WindowStyle Hidden"]


def assert_foreground_only(paths: list[str | Path]) -> None:
    violations: list[str] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        text = p.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_FOREGROUND_TOKENS:
            if token in text:
                violations.append(f"{p}:{token}")
    if violations:
        raise ValueError("foreground-only violation: " + ", ".join(violations))


def require_review_pass(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PSR review pass missing: {p}")
