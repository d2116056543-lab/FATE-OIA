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
    (out / "visual_schema.jsonl").write_text(json.dumps({"method": "DIVA-CAF-OIA V2", "fields": ["action", "reason", "factor"]}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
