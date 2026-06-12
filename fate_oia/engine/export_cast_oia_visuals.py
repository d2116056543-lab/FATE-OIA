from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_placeholder_case(output_dir: str | Path, file_name: str = "case_000") -> None:
    out = Path(output_dir) / "visuals" / file_name
    out.mkdir(parents=True, exist_ok=True)
    for name in ["original.jpg", "evidence_forward.png", "evidence_right.png", "evidence_top_reason_0.png"]:
        (out / name).write_bytes(b"")
    (out / "action_set_table.json").write_text(json.dumps({"top_action_sets": []}, indent=2), encoding="utf-8")
    (out / "graph_edges.json").write_text(json.dumps({"edges": []}, indent=2), encoding="utf-8")
    (out / "deletion_audit.json").write_text(json.dumps({"available": False}, indent=2), encoding="utf-8")
    (out / "report.html").write_text("<html><body>CAST-OIA visual report placeholder</body></html>", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    export_placeholder_case(args.output_dir)


if __name__ == "__main__":
    main()
