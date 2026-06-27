from __future__ import annotations

from pathlib import Path

from fate_oia.acpr_interactflow.artifacts import write_json


def render_case(case: dict, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "case_source.json", case)
    (out / "report.html").write_text("<html><body><h1>ACPR-InteractFlow++ case</h1></body></html>", encoding="utf-8")

