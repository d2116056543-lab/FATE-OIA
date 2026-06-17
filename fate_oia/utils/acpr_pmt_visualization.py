from __future__ import annotations

import json
from pathlib import Path


def export_chain_case(out_dir: Path, case: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chain.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.html").write_text("<html><body><pre>" + json.dumps(case, ensure_ascii=False, indent=2) + "</pre></body></html>", encoding="utf-8")
