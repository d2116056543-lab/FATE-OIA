from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.utils.care_act_artifacts import append_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    append_jsonl(out / "care_act_visuals_index.jsonl", {"status": "schema_only", "note": "Use trained checkpoint attribution audit for real heatmaps."})


if __name__ == "__main__":
    main()
