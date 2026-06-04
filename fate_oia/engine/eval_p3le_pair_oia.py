from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize a completed P3LE-PAIR-OIA V1 run.")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    best = json.loads((run_dir / "metrics_best_test.json").read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
