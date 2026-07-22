from __future__ import annotations

from pathlib import Path
import json


def export_precise_cases(output_dir: str | Path, cases: list[dict]) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        required = {"file_name", "action", "reason", "evidence", "counterfactual"}
        missing = required - set(case)
        if missing:
            raise ValueError(f"PRECISE case is missing required fields: {sorted(missing)}")
        (root / f"case_{index:03d}.json").write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")
