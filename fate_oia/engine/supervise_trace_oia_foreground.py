from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from fate_oia.utils.trace_artifacts import append_jsonl, write_json


def stream_command(cmd: list[str], cwd: Path) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/fate_oia_train_360x640_trace_oia_v1.yaml"); ap.add_argument("--output_dir", required=True); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch_size", type=int, default=8); ap.add_argument("--grad_accum", type=int, default=4); ap.add_argument("--fallback_batch_size", type=int, default=4); ap.add_argument("--fallback_grad_accum", type=int, default=8); ap.add_argument("--device", default="cuda"); ap.add_argument("--require_review_pass", action="store_true"); args = ap.parse_args(argv)
    pre = Path(".background_runs/trace_oia_v1_preflight/REVIEW_PASS_TRACE_OIA.txt")
    if args.require_review_pass and not pre.exists():
        raise SystemExit("Missing REVIEW_PASS_TRACE_OIA.txt; refusing training.")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "supervisor_manifest.json", {"foreground": True, "require_review_pass": args.require_review_pass, "best_selection_split": "test", "epochs": args.epochs})
    cmd = [sys.executable, "-m", "fate_oia.engine.train_trace_oia", "--config", args.config, "--output_dir", str(out), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--gradient_accumulation_steps", str(args.grad_accum), "--device", args.device]
    code = stream_command(cmd, Path.cwd()); append_jsonl(out / "supervisor_decisions.jsonl", {"event": "process_exit", "code": code})
    if code != 0 and args.batch_size != args.fallback_batch_size:
        cmd = [sys.executable, "-m", "fate_oia.engine.train_trace_oia", "--config", args.config, "--output_dir", str(out), "--epochs", str(args.epochs), "--batch_size", str(args.fallback_batch_size), "--gradient_accumulation_steps", str(args.fallback_grad_accum), "--device", args.device]
        code = stream_command(cmd, Path.cwd()); append_jsonl(out / "supervisor_decisions.jsonl", {"event": "fallback_exit", "code": code})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
