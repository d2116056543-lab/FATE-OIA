from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


GIT_LS_REMOTE_CONTRACT = "git ls-remote"


def run_capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()


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
    ap.add_argument("--reference_checkpoint", default="")
    args = ap.parse_args()
    print(json.dumps({"event": "heartbeat", "stage": "preflight", "fallback_ladder": [[5, 6], [4, 8], [3, 10], [2, 15]]}), flush=True)
    local_head = run_capture(["git", "rev-parse", "HEAD"])
    remote_ref = run_capture(["git", "ls-remote", "github", "refs/heads/acpr_pace_v1"])
    if remote_ref and not remote_ref.startswith(local_head):
        raise SystemExit(f"local HEAD {local_head} does not match github branch: {remote_ref}")
    if args.require_review_pass:
        rp = Path(".background_runs/acpr_pace_v1_preflight/REVIEW_PASS_ACPR_PACE_V1.txt")
        if not rp.exists():
            raise SystemExit(f"missing review pass: {rp}")
        if local_head not in rp.read_text(encoding="utf-8", errors="ignore"):
            raise SystemExit("review pass is not bound to current HEAD")
    signal_dir = Path(args.output_dir) / "pace_signal_audit"
    signal_cmd = [
        sys.executable, "-m", "fate_oia.engine.audit_acpr_pace_signal",
        "--config", args.config,
        "--checkpoint", args.reference_checkpoint,
        "--output_dir", str(signal_dir),
        "--device", args.device,
    ]
    print(json.dumps({"event": "audit_acpr_pace_signal", "cmd": signal_cmd}), flush=True)
    signal_rc = subprocess.call(signal_cmd)
    if signal_rc != 0:
        raise SystemExit(signal_rc)
    cmd = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_oia",
        "--config", args.config,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ]
    print(json.dumps({"event": "heartbeat", "stage": "train", "cmd": cmd}), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
