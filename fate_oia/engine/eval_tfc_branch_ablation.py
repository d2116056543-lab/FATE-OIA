from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="tfc_branch_ablation_stub.json")
    args = ap.parse_args()
    Path(args.output).write_text(json.dumps({"available": True, "note": "Use epoch action_branch_metrics.json for TFC branch ablation."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
