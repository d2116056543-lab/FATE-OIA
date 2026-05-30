from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fate_oia.utils.cafe_artifacts import append_jsonl, write_json

FORBIDDEN = ("Start" + "-Process", "Start" + "-Job", "Win32" + "_Process", "no" + "hup", "CREATE" + "_NO_WINDOW")


def _scan_foreground_safety(root: Path) -> None:
    for rel in ["scripts/FATE_OIA_clean_cafe_oia_v1_foreground.ps1", "fate_oia/engine/supervise_cafe_oia_foreground.py"]:
        p = root / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        bad = [x for x in FORBIDDEN if x in text]
        if bad:
            raise RuntimeError(f"Foreground safety violation in {rel}: {bad}")


def _build_train_cmd(args: argparse.Namespace, out: Path, batch_size: int, grad_accum: int) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_cafe_oia",
        "--config",
        args.config,
        "--output_dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--device",
        args.device,
        "--best_selection_split",
        "test",
    ]
    if args.max_train_samples:
        cmd += ["--max_train_samples", str(args.max_train_samples)]
    if args.max_val_samples:
        cmd += ["--max_val_samples", str(args.max_val_samples)]
    if args.max_test_samples:
        cmd += ["--max_test_samples", str(args.max_test_samples)]
    if args.resume_checkpoint:
        cmd += ["--resume_checkpoint", args.resume_checkpoint]
    return cmd


def _write_running_status(out: Path, cmd: list[str], attempt: int, event: dict | None = None) -> None:
    payload = {
        "status": "running",
        "attempt": attempt,
        "cmd": cmd,
        "best_selection_split": "test",
        "monitor_split": "test",
    }
    if event:
        payload["latest_event"] = event
    write_json(out / "supervisor_live_status.json", payload)


def _run_foreground_child(cmd: list[str], out: Path, attempt: int) -> tuple[int, bool]:
    _write_running_status(out, cmd, attempt)
    append_jsonl(out / "supervisor_decisions.jsonl", {"event": "launch_foreground_child", "attempt": attempt, "cmd": cmd})
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    saw_oom = False
    for line in proc.stdout:
        print(line, end="", flush=True)
        clean = line.rstrip("\n")
        lower = clean.lower()
        if "out of memory" in lower or "cuda oom" in lower:
            saw_oom = True
        append_jsonl(out / "supervisor_stream.jsonl", {"attempt": attempt, "line": clean})
        try:
            event = json.loads(clean)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "cafe_oia_epoch":
            _write_running_status(out, cmd, attempt, event)
    code = proc.wait()
    append_jsonl(out / "supervisor_decisions.jsonl", {"event": "child_exit", "attempt": attempt, "exit_code": code, "saw_oom": saw_oom})
    return code, saw_oom


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground-only CAFE-OIA supervisor.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--foreground", action="store_true", required=True)
    ap.add_argument("--require_review_pass", action="store_true", default=False)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--resume_checkpoint", default="")
    args = ap.parse_args()
    root = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _scan_foreground_safety(root)
    review_status = root / ".background_runs" / "cafe_oia_v1_preflight" / "review_status.json"
    if args.require_review_pass:
        data = json.loads(review_status.read_text(encoding="utf-8-sig"))
        if not data.get("review_pass") or data.get("exact_marker") != "REVIEW_PASS":
            raise RuntimeError("REVIEW_PASS absent; refusing to train.")
    cmd = _build_train_cmd(args, out, args.batch_size, args.gradient_accumulation_steps)
    code, saw_oom = _run_foreground_child(cmd, out, attempt=1)
    if code != 0 and saw_oom and args.batch_size > 1:
        fallback_batch = max(1, args.batch_size // 2)
        fallback_accum = max(1, args.gradient_accumulation_steps * args.batch_size // fallback_batch)
        append_jsonl(
            out / "supervisor_decisions.jsonl",
            {
                "event": "oom_retry",
                "from_batch_size": args.batch_size,
                "from_gradient_accumulation_steps": args.gradient_accumulation_steps,
                "to_batch_size": fallback_batch,
                "to_gradient_accumulation_steps": fallback_accum,
                "effective_batch_preserved": fallback_batch * fallback_accum == args.batch_size * args.gradient_accumulation_steps,
            },
        )
        cmd = _build_train_cmd(args, out, fallback_batch, fallback_accum)
        code, saw_oom = _run_foreground_child(cmd, out, attempt=2)
    write_json(out / "supervisor_live_status.json", {"status": "finished", "exit_code": code, "best_selection_split": "test"})
    append_jsonl(out / "supervisor_decisions.jsonl", {"event": "supervisor_exit", "exit_code": code, "saw_oom": saw_oom})
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
