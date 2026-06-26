from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_evidence_chain(action: str, evidence_slot: str, predicate: str, reason: str, patch_xy: tuple[int, int]) -> dict[str, Any]:
    return {
        "action": action,
        "evidence_slot": evidence_slot,
        "predicate": predicate,
        "reason": reason,
        "patch_xy": [int(patch_xy[0]), int(patch_xy[1])],
    }


def write_evidence_report(path: str | Path, chains: list[dict[str, Any]]) -> None:
    path = Path(path)
    rows = "\n".join(
        f"<tr><td>{c.get('action')}</td><td>{c.get('evidence_slot')}</td><td>{c.get('predicate')}</td><td>{c.get('reason')}</td><td>{c.get('patch_xy')}</td></tr>"
        for c in chains
    )
    path.write_text(
        "<html><body><table><tr><th>action</th><th>evidence</th><th>predicate</th><th>reason</th><th>patch</th></tr>"
        + rows
        + "</table></body></html>",
        encoding="utf-8",
    )


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
