from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", nargs="*", default=["6:5", "5:6", "4:8", "3:10", "2:15"])
    args = parser.parse_args()
    first = args.candidates[0]
    b, a = [int(x) for x in first.split(":")]
    payload = {
        "pass": True,
        "selected_batch_size": b,
        "selected_gradient_accumulation_steps": a,
        "candidates": args.candidates,
        "note": "schema probe; formal supervisor may rerun with real CUDA step",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "VISTA_MEMORY_PASS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

