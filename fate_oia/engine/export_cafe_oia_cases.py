from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    run = Path(args.run_dir)
    output = Path(args.output) if args.output else run / "cafe_case_index.jsonl"
    rows = []
    for p in sorted(run.glob("epoch_*/metrics_summary.json")):
        rows.append({"epoch_dir": str(p.parent), "metrics_summary": str(p)})
    output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

