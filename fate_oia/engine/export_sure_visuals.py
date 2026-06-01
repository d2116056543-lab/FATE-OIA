from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.utils.sure_artifacts import append_jsonl, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SURE visualization schema records.")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for idx in range(args.max_samples):
        append_jsonl(out / "sure_visuals_index.jsonl", {"sample_index": idx, "checkpoint": args.checkpoint, "schema": "sure_relation_attention"})
    write_json(out / "visual_export_summary.json", {"records": args.max_samples, "checkpoint": args.checkpoint})


if __name__ == "__main__":
    main()
