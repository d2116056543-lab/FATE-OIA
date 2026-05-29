from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / ".background_runs"


def write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_checked(cmd: list[str], cwd: Path, log: Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n")
            f.write(proc.stdout + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")


def append_progress(text: str) -> None:
    path = Path(r"E:\sbw\FATE_Drive\progress.md")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + text.rstrip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Supervise RunC integrated specialist launch.")
    ap.add_argument("--launch_training", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--review_pass_file", default=str(ROOT / ".background_runs" / "runc_integrated_v1_preflight" / "review_pass.txt"))
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    preflight = ROOT / ".background_runs" / "runc_integrated_v1_preflight"
    decisions = preflight / "supervisor_decisions.jsonl"
    write_jsonl(decisions, {"event": "supervisor_start", "time": datetime.now().isoformat(), "args": vars(args)})
    files = [str(p) for d in [ROOT / "fate_oia" / "engine", ROOT / "fate_oia" / "models", ROOT / "fate_oia" / "losses"] for p in d.glob("*.py")]
    run_checked([r"E:\Anaconda\envs\sbw39\python.exe", "-m", "py_compile", *files], ROOT, preflight / "py_compile.log")
    run_checked([r"E:\Anaconda\envs\sbw39\python.exe", "-m", "pytest", "tests/test_runc_integrated_specialist.py", "-q"], ROOT, preflight / "pytest.log")
    run_checked(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts_runc/run_c_eval_test.ps1", "-Output", str(preflight / "parity_eval_supervisor.json")], ROOT, preflight / "parity.log")
    parity = json.loads((preflight / "parity_eval_supervisor.json").read_text(encoding="utf-8"))
    if not parity.get("passed"):
        raise RuntimeError("Run C parity did not pass; refusing training launch")
    if args.launch_training:
        review = Path(args.review_pass_file)
        if not review.exists() or "REVIEW_PASS" not in review.read_text(encoding="utf-8", errors="ignore"):
            raise RuntimeError(f"REVIEW_PASS missing: {review}")
        if not args.output_dir:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = RUN_ROOT / f"runc_integrated_specialist_v1_{stamp}"
        else:
            out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stdout = out_dir / "train_stdout.log"
        stderr = out_dir / "train_stderr.log"
        cmd = [
            r"E:\Anaconda\envs\sbw39\python.exe", "-m", "fate_oia.engine.train_runc_integrated_specialist",
            "--output_dir", str(out_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--max_train_samples", str(args.max_train_samples),
            "--max_val_samples", str(args.max_val_samples),
            "--max_test_samples", str(args.max_test_samples),
            "--device", args.device,
        ]
        out_f = stdout.open("w", encoding="utf-8")
        err_f = stderr.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out_f, stderr=err_f, text=True)
        payload = {"event": "training_launched", "time": datetime.now().isoformat(), "pid": proc.pid, "output_dir": str(out_dir), "stdout": str(stdout), "stderr": str(stderr), "cmd": cmd}
        write_jsonl(out_dir / "supervisor_decisions.jsonl", payload)
        write_jsonl(decisions, payload)
        append_progress(f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - RunC integrated specialist background launch\n- Worktree: {ROOT}\n- PID: {proc.pid}\n- Output: {out_dir}\n- Stdout: {stdout}\n- Stderr: {stderr}\n- Launch gate: py_compile PASS, pytest PASS, parity PASS, REVIEW_PASS present.")
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
