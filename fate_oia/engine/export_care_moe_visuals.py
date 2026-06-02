from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.utils.care_moe_artifacts import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir) / "care_moe_visuals_index.jsonl"
    write_json(out, {"status": "schema_ready", "note": "Visual export consumes trained CARE-MoE diagnostics and evidence anchors."})
    print(out)


if __name__ == "__main__":
    main()
