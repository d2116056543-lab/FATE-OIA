from __future__ import annotations

from pathlib import Path


def export_precise_cases(output_dir: str | Path, cases: list[dict]) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        (root / f"case_{index:03d}.json").write_text(str(case), encoding="utf-8")
