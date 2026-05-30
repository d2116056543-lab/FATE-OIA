from __future__ import annotations

import json
from pathlib import Path

from fate_oia.utils.cafe_artifacts import append_jsonl, write_json


def test_artifact_schema_complete(tmp_path: Path):
    write_json(tmp_path / "run_manifest.json", {"best_selection_split": "test"})
    append_jsonl(tmp_path / "supervisor_decisions.jsonl", {"event": "x"})
    assert json.loads((tmp_path / "run_manifest.json").read_text())["best_selection_split"] == "test"
    assert (tmp_path / "supervisor_decisions.jsonl").read_text().strip()

