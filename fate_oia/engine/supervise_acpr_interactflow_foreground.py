from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _check_review_pass_bound_to_head() -> None:
    review_path = Path(".background_runs/acpr_interactflow_pp_v1_preflight/REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt")
    if not review_path.exists():
        raise SystemExit("Missing REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    local_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if review.get("git_head") != local_head:
        raise SystemExit(f"Stale REVIEW_PASS: review git_head={review.get('git_head')} local HEAD={local_head}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty:
        raise SystemExit("Worktree is dirty; commit code-only changes and rerun preflight before full train.")
    remote = subprocess.check_output(["git", "ls-remote", "github", "refs/heads/acpr_interactflow_pp_v1"], text=True).strip()
    remote_head = remote.split()[0] if remote else ""
    if remote_head != local_head:
        raise SystemExit(f"GitHub branch HEAD mismatch: remote={remote_head} local={local_head}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require_review_pass", action="store_true")
    args = parser.parse_args()
    if args.require_review_pass:
        _check_review_pass_bound_to_head()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_interactflow_psi",
        "--config",
        args.config,
        "--output_dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "--device",
        args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ]
    print("foreground_exec " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
