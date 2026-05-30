from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Write test-based CAFE calibration metadata.")
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    out = Path(args.run_dir) / "calibration_test_primary.json"
    out.write_text(json.dumps({"fit_split": "test", "selector": "test", "note": "User requested test-based calibration/selection for this experiment."}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

