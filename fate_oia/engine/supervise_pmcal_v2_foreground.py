from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_pmcal_v2.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=9)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--review_pass_path", default=".background_runs/acpr_pmcal_v2_preflight/REVIEW_PASS_PMCalV2.txt")
    args = ap.parse_args()
    if args.require_review_pass and not Path(args.review_pass_path).exists():
        raise SystemExit(f"missing review pass: {args.review_pass_path}")
    cmd = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_pmcal_v2_oia",
        "--config", args.config,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--device", args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
        "--require_review_pass",
        "--review_pass_path", args.review_pass_path,
    ]
    proc = subprocess.Popen(cmd)
    code = proc.wait()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
