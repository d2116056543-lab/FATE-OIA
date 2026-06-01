from __future__ import annotations

from pathlib import Path


FORBIDDEN_SUPERVISOR_PATTERNS = [
    "Start-Process",
    "Start-Job",
    "Win32_Process",
    "Invoke-WmiMethod",
    "nohup",
]


def assert_no_forbidden_supervisor_patterns(paths: list[str | Path]) -> None:
    hits: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_SUPERVISOR_PATTERNS:
            if pattern in text:
                hits.append(f"{path}:{pattern}")
    if hits:
        raise RuntimeError("Forbidden non-foreground launch pattern found: " + ", ".join(hits))


def assert_test_only_manifest(manifest: dict) -> None:
    if manifest.get("eval_splits") != ["test"]:
        raise RuntimeError(f"SURE run must be test-only, got eval_splits={manifest.get('eval_splits')}")
    if manifest.get("uses_val"):
        raise RuntimeError("SURE run marked uses_val=true")
    if manifest.get("uses_feature_cache"):
        raise RuntimeError("SURE direct-image run must not use feature cache")
