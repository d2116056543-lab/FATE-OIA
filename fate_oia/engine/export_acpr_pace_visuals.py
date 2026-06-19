from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "available": True,
        "source_epoch_dir": args.epoch_dir,
        "exports": ["action_reason_predicate_contribution_chain", "predicate_patch_positive_negative"],
    }
    (out / "pace_visual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
