from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FALLBACK_LADDER = [(5, 6), (4, 8), (3, 10), (2, 16)]
EMERGENCY_FALLBACK = (1, 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--grad_accum", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--output_dir", default=".background_runs/cast_oia_v1_full")
    args = ap.parse_args()
    if args.require_review_pass:
        # Keep this literal command visible for audit: git rev-parse HEAD
        current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        review_candidates = [
            Path(".background_runs/cast_oia_v1_preflight_postcommit/REVIEW_PASS_CAST_OIA_V1.txt"),
            Path(".background_runs/cast_oia_v1_preflight/REVIEW_PASS_CAST_OIA_V1.txt"),
        ]
        review = next((p for p in review_candidates if p.exists()), None)
        if review is None:
            raise SystemExit("missing required review pass: REVIEW_PASS_CAST_OIA_V1.txt")
        review_text = review.read_text(encoding="utf-8", errors="ignore")
        if current_head not in review_text:
            raise SystemExit(f"review pass {review} is not bound to current HEAD {current_head}")
    candidates = [(args.batch_size, args.grad_accum)] + [x for x in FALLBACK_LADDER if x != (args.batch_size, args.grad_accum)] + [EMERGENCY_FALLBACK]
    last = None
    for batch, accum in candidates:
        cmd = [
            sys.executable, "-u", "-m", "fate_oia.engine.train_cast_oia",
            "--config", "configs/fate_oia_train_360x640_cast_oia_v1.yaml",
            "--output_dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(batch),
            "--gradient_accumulation_steps", str(accum),
            "--device", args.device,
            "--test_only",
            "--no_feature_cache",
            "--require_no_token_compression",
        ]
        print("CAST_FOREGROUND_COMMAND " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd)
        proc.wait()
        if proc.returncode == 0:
            return
        last = proc.returncode
        print(f"CAST_RUN_FAILED returncode={proc.returncode}; trying next fallback if available", flush=True)
    raise SystemExit(last or 1)


if __name__ == "__main__":
    main()
