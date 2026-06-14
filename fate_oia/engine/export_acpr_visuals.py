from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.utils.acpr_artifacts import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--file_name", default="case.jpg")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "matched_counterfactual_comparison.json", {"file_name": args.file_name, "available": False})
    write_json(out / "predicate_attention_summary.json", {"available": False})
    write_json(out / "report.json", {"available": True})


if __name__ == "__main__":
    main()
