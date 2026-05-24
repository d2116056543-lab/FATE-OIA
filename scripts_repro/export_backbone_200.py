#!/usr/bin/env python3
"""Export the SNNA stage-1 DINO checkpoint to the README path.

SNNA's classifier command expects `ckp/backbone_200.pth`. `main_dino.py`
writes the latest DINO checkpoint as `checkpoint.pth` inside the output
directory. Because `multi_label_train.py` loads checkpoint key `teacher`, the
whole DINO checkpoint is a valid `backbone_200.pth`.
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino_checkpoint", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.dino_checkpoint.exists():
        raise FileNotFoundError(args.dino_checkpoint)
    if args.output_path.exists() and not args.force:
        raise FileExistsError(f"{args.output_path} already exists; pass --force to overwrite")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.dino_checkpoint, args.output_path)

    summary = {
        "source": str(args.dino_checkpoint.resolve()),
        "output": str(args.output_path.resolve()),
        "bytes": args.output_path.stat().st_size,
        "note": "Whole DINO checkpoint copied because multi_label_train.py loads checkpoint_key='teacher'.",
    }
    summary_path = args.output_path.with_suffix(args.output_path.suffix + ".export_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
