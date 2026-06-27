from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.explain.acpr_interactflow_atlas import build_atlas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--intervention")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    case_dirs = sorted([p for p in input_dir.glob("case_*") if p.is_dir()])
    manifest = build_atlas(
        case_dirs,
        args.output,
        metrics_path=args.metrics,
        intervention_path=args.intervention,
    )
    if manifest["case_count"] == 0:
        raise SystemExit(f"No case_* directories found under {input_dir}")


if __name__ == "__main__":
    main()
