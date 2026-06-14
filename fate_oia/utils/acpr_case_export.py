from __future__ import annotations

from pathlib import Path

from fate_oia.utils.acpr_artifacts import write_json


def export_case_stub(output_dir: str | Path, file_name: str, payload: dict) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{Path(file_name).stem}_case.json", payload)
