from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "available": False,
        "required_maps": ["raw_image", "action_attention", "reason_attention", "predicate_attention", "vista_delta_map", "predicate_gate_map"],
    }
    (out / "vista_visual_report_schema.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

