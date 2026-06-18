from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch_dir", required=True)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    epoch_dir = Path(args.epoch_dir)
    out_dir = Path(args.output_dir) if args.output_dir else epoch_dir / "seca_visual_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_path = epoch_dir / "seca_evidence_chains.jsonl"
    rows = []
    if chain_path.exists():
        for line in chain_path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    (out_dir / "seca_evidence_chains.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    body = ["<html><body><h1>ACPR-SECA Visual Evidence Export</h1><table border='1'>",
            "<tr><th>category</th><th>file</th><th>top action</th><th>top reason</th><th>null attention</th></tr>"]
    for item in rows[:200]:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('category','')))}</td>"
            f"<td>{html.escape(str(item.get('file_name','')))}</td>"
            f"<td>{html.escape(str(item.get('top_action','')))}</td>"
            f"<td>{html.escape(str(item.get('top_reason','')))}</td>"
            f"<td>{html.escape(str(item.get('null_attention','')))}</td>"
            "</tr>"
        )
    body.append("</table></body></html>")
    (out_dir / "seca_evidence_report.html").write_text("\n".join(body), encoding="utf-8")
    print(str(out_dir / "seca_evidence_report.html"), flush=True)


if __name__ == "__main__":
    main()
