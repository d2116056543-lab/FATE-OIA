from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    out_dir = Path(".background_runs/acpr_ntmcal_v1_360x640_testprimary")
    out_dir.mkdir(parents=True, exist_ok=True)
    review = Path(".background_runs/acpr_ntmcal_v1_preflight/REVIEW_PASS_ACPR_NTMCAL_V1.txt")
    if not review.exists():
        raise SystemExit(f"missing review pass: {review}")
    (out_dir / "supervisor_live_status.json").write_text(json.dumps({"alive": True, "event": "launch"}, indent=2), encoding="utf-8")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_ntmcal_oia",
        "--config",
        "configs/fate_oia_train_360x640_acpr_ntmcal_v1.yaml",
        "--output_dir",
        str(out_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--num_workers",
        str(args.num_workers),
        "--test_only",
        "--best_selection_split",
        "test",
        "--best_selection_metric",
        "joint_test_score",
        "--no_feature_cache",
        "--token_compression",
        "none",
        "--require_no_token_compression",
        "--require_review_pass",
        str(review),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
