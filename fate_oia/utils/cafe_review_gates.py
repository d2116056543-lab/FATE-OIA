from __future__ import annotations

from pathlib import Path


FORBIDDEN_FOREGROUND_TOKENS = ("Start-Process", "Start-Job", "Win32_Process", "Invoke-WmiMethod", "nohup", "-WindowStyle Hidden")

FORBIDDEN_PLACEHOLDERS = (
    "diagnostic placeholder",
    "case export placeholder",
    "computed_proxy",
    "target_deleted_reason = reason_logits - torch.relu",
    'metrics_test_calibrated.json", test_stats["metrics"]',
    'calibration_params.json", {"fit_split": "test"',
    "pass  # TODO",
    "TODO: implement",
)


def scan_forbidden_tokens(paths: list[str | Path], tokens: tuple[str, ...]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for path in paths:
        p = Path(path)
        if not p.exists() or p.is_dir():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        found = [tok for tok in tokens if tok in text]
        if found:
            hits[str(p)] = found
    return hits


def require_review_pass(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"REVIEW_PASS_CAFE_V2 is required before training: {p}")
