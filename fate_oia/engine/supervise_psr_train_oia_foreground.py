from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def append_jsonl(path: Path, row: dict) -> None:
    path.write_text(path.read_text(encoding="utf-8") + json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def run_stream(cmd: list[str], cwd: Path, log_path: Path | None = None) -> int:
    print(json.dumps({"event": "foreground_command_start", "cmd": cmd}, ensure_ascii=False), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    log_handle = log_path.open("a", encoding="utf-8") if log_path else None
    for line in proc.stdout:
        text = line.rstrip()
        print(text, flush=True)
        if log_handle:
            log_handle.write(text + "\n")
            log_handle.flush()
    code = int(proc.wait())
    if log_handle:
        log_handle.write(json.dumps({"event": "foreground_command_exit", "exit_code": code}, ensure_ascii=False) + "\n")
        log_handle.close()
    return code


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground PSR-Train OIA V1 supervisor.")
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_psr_train_oia_v1.yaml")
    ap.add_argument("--output_root", default=".background_runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--require_audit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume_output_dir", default="")
    ap.add_argument("--resume_psr_train_checkpoint", default="")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.resume_output_dir) if args.resume_output_dir else Path(args.output_root) / f"psr_train_oia_v1_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = output_dir / "supervisor_decisions.jsonl"
    if not decisions.exists():
        decisions.write_text("", encoding="utf-8")
    resume_checkpoint = Path(args.resume_psr_train_checkpoint) if args.resume_psr_train_checkpoint else output_dir / "checkpoint_latest.pth"
    resume_active = bool(args.resume_output_dir)
    append_jsonl(decisions, {
        "event": "supervisor_resume_start" if resume_active else "supervisor_start",
        "output_dir": str(output_dir),
        "foreground": True,
        "background_forbidden": True,
        "resume_active": resume_active,
        "resume_psr_train_checkpoint": str(resume_checkpoint) if resume_active else "",
    })
    if args.require_audit:
        audit_cmd = [
            sys.executable,
            "-m",
            "fate_oia.engine.audit_psr_train_oia_implementation",
            "--config",
            args.config,
            "--output_dir",
            str(output_dir / "preflight_audit"),
        ]
        code = run_stream(audit_cmd, root, output_dir / "audit.log")
        if code != 0:
            append_jsonl(decisions, {"event": "audit_failed", "exit_code": code})
            raise SystemExit(code)
    train_cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_psr_train_oia",
        "--config",
        args.config,
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--num_workers",
        str(args.num_workers),
        "--max_train_samples",
        str(args.max_train_samples),
        "--max_test_samples",
        str(args.max_test_samples),
    ]
    if resume_active:
        if not resume_checkpoint.exists():
            append_jsonl(decisions, {"event": "resume_checkpoint_missing", "path": str(resume_checkpoint)})
            raise SystemExit(2)
        train_cmd.extend(["--resume_psr_train_checkpoint", str(resume_checkpoint)])
    code = run_stream(train_cmd, root, output_dir / "train.log")
    append_jsonl(decisions, {"event": "training_exit", "exit_code": code})
    raise SystemExit(code)


if __name__ == "__main__":
    main()
