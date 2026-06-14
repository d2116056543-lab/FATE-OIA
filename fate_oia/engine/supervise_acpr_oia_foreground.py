from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_stream(cmd: list[str]) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    last = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        last = time.time()
    code = proc.wait()
    if time.time() - last > 1800:
        print(json.dumps({"event": "acpr_supervisor_stall_checked", "seconds_since_output": time.time() - last}), flush=True)
    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    preflight = out / "supervisor_preflight"
    audit_cmd = [sys.executable, "-m", "fate_oia.engine.audit_acpr_oia_implementation", "--config", args.config, "--output_dir", str(preflight), "--device", args.device, "--write_review_pass"]
    if run_stream(audit_cmd) != 0:
        raise SystemExit("ACPR audit failed; full training blocked")
    pass_file = preflight / "REVIEW_PASS_ACPR_OIA_V1.txt"
    if args.require_review_pass and not pass_file.exists():
        raise SystemExit("Missing REVIEW_PASS_ACPR_OIA_V1.txt")
    smoke_cmd = [sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_oia", "--config", args.config, "--output_dir", str(out / "smoke"), "--epochs", "1", "--batch_size", "1", "--gradient_accumulation_steps", "2", "--max_train_samples", "4", "--max_test_samples", "4", "--device", args.device, "--test_only", "--no_feature_cache", "--require_no_token_compression"]
    if run_stream(smoke_cmd) != 0:
        raise SystemExit("ACPR smoke failed; full training blocked")
    fallback_ladder = [(args.batch_size, args.gradient_accumulation_steps), (5, 6), (4, 8), (3, 11)]
    last_error = None
    for batch, accum in fallback_ladder:
        train_cmd = [sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_oia", "--config", args.config, "--output_dir", str(out), "--epochs", str(args.epochs), "--batch_size", str(batch), "--gradient_accumulation_steps", str(accum), "--device", args.device, "--test_only", "--no_feature_cache", "--require_no_token_compression"]
        print(json.dumps({"event": "acpr_supervisor_launch", "batch_size": batch, "grad_accum": accum, "fallback_ladder": fallback_ladder}), flush=True)
        code = run_stream(train_cmd)
        if code == 0:
            goal = out / "GOAL_COMPLETED_ACPR_OIA_V1.json"
            if goal.exists():
                print(json.dumps({"event": "acpr_supervisor_completed", "goal": str(goal)}), flush=True)
                return
            last_error = "train exited 0 but goal file missing"
            break
        last_error = f"train exited {code}"
        print(json.dumps({"event": "acpr_supervisor_retry_after_failure", "error": last_error}), flush=True)
    raise SystemExit(last_error or "ACPR training failed")


if __name__ == "__main__":
    main()
