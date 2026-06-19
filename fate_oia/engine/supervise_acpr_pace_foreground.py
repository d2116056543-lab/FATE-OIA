from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_pace_v1.yaml")
    ap.add_argument("--output_dir", default=".background_runs/acpr_pace_v1_full")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_oia",
        "--config",
        args.config,
        "--output_dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ]
    if args.require_review_pass:
        import pathlib
        rp = pathlib.Path(".background_runs/acpr_pace_v1_preflight/REVIEW_PASS_ACPR_PACE_V1.txt")
        if not rp.exists():
            raise SystemExit(f"missing review pass: {rp}")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
