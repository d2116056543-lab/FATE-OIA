from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_oia_v1.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    if args.require_review_pass and not Path(".background_runs/acpr_oia_v1_preflight/REVIEW_PASS_ACPR_OIA_V1.txt").exists():
        raise SystemExit("Missing REVIEW_PASS_ACPR_OIA_V1.txt")
    cmd = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_oia",
        "--config", args.config,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--device", args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ]
    print("ACPR_FOREGROUND_CMD " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
