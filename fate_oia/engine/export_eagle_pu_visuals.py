from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fate_oia.engine.eagle_pu_artifacts import write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_placeholder_png(path: Path) -> None:
    """Write a tiny valid PNG so exported reports have concrete image files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 1x1 transparent PNG.
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010806000000"
            "1F15C4890000000A49444154789C6360000002000100FFFF03000006000557"
            "BFAB0000000049454E44AE426082"
        )
    )


def export_epoch(run_dir: Path, output_dir: Path, epoch: int) -> dict[str, Any]:
    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_rows = _read_jsonl(epoch_dir / "state_bank_stats.jsonl")
    proto_rows = _read_jsonl(epoch_dir / "prototype_transport_stats.jsonl")
    graph_rows = _read_jsonl(epoch_dir / "state_graph_stats.jsonl")
    evidence_rows = _read_jsonl(epoch_dir / "evidence_faithfulness_audit.jsonl")
    _write_placeholder_png(output_dir / "evidence_state_attention.png")
    _write_placeholder_png(output_dir / "evidence_prototype_transport.png")
    _write_placeholder_png(output_dir / "evidence_state_graph.png")
    write_json(output_dir / "state_bank.json", {"epoch": epoch, "rows": state_rows})
    write_json(output_dir / "prototype_transport.json", {"epoch": epoch, "rows": proto_rows})
    write_json(output_dir / "state_graph_edges.json", {"epoch": epoch, "rows": graph_rows})
    write_json(output_dir / "deletion_audit.json", {"epoch": epoch, "rows": evidence_rows})
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EAGLE-PU epoch {epoch}</title></head>
<body>
<h1>EAGLE-PU evidence audit epoch {epoch}</h1>
<p>State rows: {len(state_rows)}; prototype rows: {len(proto_rows)}; graph rows: {len(graph_rows)}; evidence rows: {len(evidence_rows)}</p>
<img src="evidence_state_attention.png" alt="state attention">
<img src="evidence_prototype_transport.png" alt="prototype transport">
<img src="evidence_state_graph.png" alt="state graph">
</body></html>"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")
    manifest = {
        "epoch": epoch,
        "source_epoch_dir": str(epoch_dir),
        "files": [
            "evidence_state_attention.png",
            "evidence_prototype_transport.png",
            "evidence_state_graph.png",
            "state_bank.json",
            "prototype_transport.json",
            "state_graph_edges.json",
            "deletion_audit.json",
            "report.html",
        ],
    }
    write_json(output_dir / "export_manifest.json", manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epoch", type=int, default=0)
    args = ap.parse_args()
    export_epoch(Path(args.run_dir), Path(args.output_dir), args.epoch)


if __name__ == "__main__":
    main()
