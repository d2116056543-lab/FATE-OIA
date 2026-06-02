from __future__ import annotations

from pathlib import Path


PASS_TOKEN = "REVIEW_PASS_CARE_MOE_OIA_V1"


def require_review_pass(root: str | Path = ".") -> Path:
    path = Path(root) / ".background_runs" / "care_moe_oia_v1_preflight" / "REVIEW_PASS_CARE_MOE_OIA_V1.txt"
    if not path.exists() or PASS_TOKEN not in path.read_text(encoding="utf-8", errors="ignore"):
        raise RuntimeError(f"Missing CARE-MoE review pass file: {path}")
    return path
