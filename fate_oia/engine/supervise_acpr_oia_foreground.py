from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def append_log(log_path: Path | None, text: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_stream(cmd: list[str], log_path: Path | None = None) -> int:
    append_log(log_path, json.dumps({"event": "acpr_supervisor_command", "cmd": cmd}, ensure_ascii=False))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    last = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        append_log(log_path, line.rstrip("\n"))
        last = time.time()
    code = proc.wait()
    if time.time() - last > 1800:
        stall = json.dumps({"event": "acpr_supervisor_stall_checked", "seconds_since_output": time.time() - last}, ensure_ascii=False)
        print(stall, flush=True)
        append_log(log_path, stall)
    append_log(log_path, json.dumps({"event": "acpr_supervisor_command_exit", "returncode": code}, ensure_ascii=False))
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
    ap.add_argument("--resume_checkpoint", default="")
    ap.add_argument("--stop_after_epochs", type=int, default=None)
    ap.add_argument("--sanity_finetune", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    supervisor_log = out / "supervisor.log"
    append_log(
        supervisor_log,
        json.dumps(
            {
                "event": "acpr_supervisor_start",
                "output_dir": str(out),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "device": args.device,
            },
            ensure_ascii=False,
        ),
    )
    preflight = out / "supervisor_preflight"
    audit_module = "fate_oia.engine.audit_acpr_fusionlite_implementation" if "fusionlite" in args.config.lower() else "fate_oia.engine.audit_acpr_oia_implementation"
    audit_cmd = [sys.executable, "-m", audit_module, "--config", args.config, "--output_dir", str(preflight), "--device", args.device, "--write_review_pass"]
    if run_stream(audit_cmd, supervisor_log) != 0:
        raise SystemExit("ACPR audit failed; full training blocked")
    pass_file = preflight / ("REVIEW_PASS_ACPR_FUSIONLITE_V1_4.txt" if "fusionlite" in args.config.lower() else "REVIEW_PASS_ACPR_OIA_V1.txt")
    if args.require_review_pass and not pass_file.exists():
        raise SystemExit("Missing REVIEW_PASS_ACPR_OIA_V1.txt")
    smoke_cmd = [sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_oia", "--config", args.config, "--output_dir", str(out / "smoke"), "--epochs", "1", "--batch_size", "1", "--gradient_accumulation_steps", "2", "--max_train_samples", "4", "--max_test_samples", "4", "--device", args.device, "--test_only", "--no_feature_cache", "--require_no_token_compression"]
    if run_stream(smoke_cmd, supervisor_log) != 0:
        raise SystemExit("ACPR smoke failed; full training blocked")
    fallback_ladder = [(args.batch_size, args.gradient_accumulation_steps), (5, 6), (4, 8), (3, 11), (2, 16)]
    last_error = None
    for batch, accum in fallback_ladder:
        train_cmd = [sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_oia", "--config", args.config, "--output_dir", str(out), "--epochs", str(args.epochs), "--batch_size", str(batch), "--gradient_accumulation_steps", str(accum), "--device", args.device, "--test_only", "--no_feature_cache", "--require_no_token_compression"]
        if args.resume_checkpoint:
            train_cmd.extend(["--resume_checkpoint", args.resume_checkpoint])
        if args.stop_after_epochs is not None:
            train_cmd.extend(["--stop_after_epochs", str(args.stop_after_epochs)])
        if args.sanity_finetune:
            train_cmd.append("--sanity_finetune")
        launch = json.dumps({"event": "acpr_supervisor_launch", "batch_size": batch, "grad_accum": accum, "fallback_ladder": fallback_ladder}, ensure_ascii=False)
        print(launch, flush=True)
        append_log(supervisor_log, launch)
        code = run_stream(train_cmd, supervisor_log)
        if code == 0:
            goal = out / "GOAL_COMPLETED_ACPR_OIA_V1.json"
            fusion_goal = out / "GOAL_COMPLETED_ACPR_FUSIONLITE_V1_4.json"
            if goal.exists() or fusion_goal.exists():
                done = json.dumps({"event": "acpr_supervisor_completed", "goal": str(goal)}, ensure_ascii=False)
                print(done, flush=True)
                append_log(supervisor_log, done)
                return
            last_error = "train exited 0 but goal file missing"
            break
        last_error = f"train exited {code}"
        retry = json.dumps({"event": "acpr_supervisor_retry_after_failure", "error": last_error}, ensure_ascii=False)
        print(retry, flush=True)
        append_log(supervisor_log, retry)
    raise SystemExit(last_error or "ACPR training failed")


if __name__ == "__main__":
    main()
