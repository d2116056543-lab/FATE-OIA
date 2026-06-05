from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_oia.utils.egcaf_visual_export import export_factor_overlay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual_factor_samples", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_root", default="")
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.visual_factor_samples).read_text(encoding="utf-8").splitlines() if x.strip()]
    for i, row in enumerate(rows[:2]):
        export_factor_overlay(row.get("image_path", ""), [row], Path(args.output_dir) / f"sample_{i}.png")


if __name__ == "__main__":
    main()
