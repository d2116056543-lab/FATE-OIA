from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export P3LE visual schema rows for later FATE-SNNA diagnostics.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {"run_dir": args.run_dir, "method": "P3LE-PAIR-OIA V1", "visuals": "schema_only"}
    (out_dir / "fate_snna_schema.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
