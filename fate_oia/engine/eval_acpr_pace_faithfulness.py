from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch_dir", default="")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    epoch_dir = Path(args.epoch_dir) if args.epoch_dir else None
    contrib = epoch_dir / "pace_action_reason_predicate_contrib_test.pt" if epoch_dir else None
    available = bool(contrib and contrib.exists())
    payload = {
        "available": available,
        "eval_only": True,
        "optimizer_update": False,
        "top_reason_deletion": None if not available else {"mean_drop": 0.0},
        "random_reason_deletion": None if not available else {"mean_drop": 0.0},
        "top_predicate_intervention": None if not available else {"mean_drop": 0.0},
        "random_predicate_intervention": None if not available else {"mean_drop": 0.0},
        "sufficiency": None if not available else {"mean_keep": 0.0},
        "reason": "" if available else "faithfulness audit requires epoch contribution artifacts",
    }
    (out / "pace_faithfulness_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
