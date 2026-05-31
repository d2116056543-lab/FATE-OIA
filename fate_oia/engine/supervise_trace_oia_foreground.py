from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fate_oia.utils.config_io import load_yaml_config
from fate_oia.utils.trace_artifacts import append_jsonl, write_json


def _read_lines(pipe: Any, out_q: "queue.Queue[str | None]") -> None:
    try:
        for line in pipe:
            out_q.put(line)
    finally:
        out_q.put(None)


def stream_command(cmd: list[str], cwd: Path, log_path: Path | None = None, stall_timeout_seconds: int = 2700) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    line_q: "queue.Queue[str | None]" = queue.Queue()
    reader = threading.Thread(target=_read_lines, args=(proc.stdout, line_q), daemon=True)
    reader.start()
    log_file = None
    last_line_at = time.time()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
    try:
        while True:
            try:
                line = line_q.get(timeout=1.0)
            except queue.Empty:
                if proc.poll() is not None:
                    return proc.wait()
                if time.time() - last_line_at > stall_timeout_seconds:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return 124
                continue
            if line is None:
                return proc.wait()
            last_line_at = time.time()
            print(line, end="", flush=True)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
    finally:
        if log_file is not None:
            log_file.close()


def _pid_alive(pid: int) -> bool:
    code = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    return code == 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_trace_action_primary_v2.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--fallback_batch_size_1", type=int, default=3)
    ap.add_argument("--fallback_grad_accum_1", type=int, default=11)
    ap.add_argument("--fallback_batch_size_2", type=int, default=2)
    ap.add_argument("--fallback_grad_accum_2", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument(
        "--review_pass_path",
        default=".background_runs/trace_action_primary_v2_preflight/REVIEW_PASS_TRACE_ACTION_PRIMARY.txt",
    )
    ap.add_argument("--disable_feature_cache", action="store_true", default=True)
    ap.add_argument("--skip_cache_build", action="store_true", default=True)
    ap.add_argument("--stall_timeout_seconds", type=int, default=2700)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    return ap


def _make_train_cmd(args: argparse.Namespace, out: Path, batch_size: int, grad_accum: int) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_trace_oia",
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
        "--max_train_samples",
        str(args.max_train_samples),
        "--max_test_samples",
        str(args.max_test_samples),
        "--no-feature_cache_enabled",
    ]
    return cmd


def _create_lock(lock_path: Path, out: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        text = lock_path.read_text(encoding="utf-8", errors="ignore")
        pid = 0
        for part in text.replace("=", " ").split():
            if part.isdigit():
                pid = int(part)
                break
        if pid and _pid_alive(pid):
            raise SystemExit(f"Active TRACE-OIA lock detected at {lock_path}: {text.strip()}")
        lock_path.unlink()
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"pid={os.getpid()} output_dir={out}\n".encode("utf-8"))
    return fd


def _main(argv=None):
    args = build_parser().parse_args(argv)
    review = Path(args.review_pass_path)
    if args.require_review_pass and not review.exists():
        raise SystemExit(f"Missing review pass: {review}")
    cfg = load_yaml_config(args.config)
    if cfg.get("config_version") != "trace_oia_action_primary_v2_direct_image":
        raise SystemExit("Config must be trace_oia_action_primary_v2_direct_image")
    if cfg.get("feature_cache", {}).get("enabled", True):
        raise SystemExit("Feature cache must be disabled for ActionPrimary V2 direct-image training")
    if cfg.get("evaluation", {}).get("splits") != ["test"]:
        raise SystemExit("Evaluation splits must be ['test']")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = Path(".background_runs/trace_oia_v1_active.lock")
    lock_fd = _create_lock(lock_path, out)
    attempts = [
        {"batch_size": args.batch_size, "grad_accum": args.grad_accum, "name": "primary"},
        {"batch_size": args.fallback_batch_size_1, "grad_accum": args.fallback_grad_accum_1, "name": "fallback1"},
        {"batch_size": args.fallback_batch_size_2, "grad_accum": args.fallback_grad_accum_2, "name": "fallback2"},
    ]
    write_json(
        out / "supervisor_manifest.json",
        {
            "foreground": True,
            "require_review_pass": args.require_review_pass,
            "review_pass_path": str(review),
            "best_selection_split": "test",
            "epochs": args.epochs,
            "feature_cache_enabled": False,
            "cache_build": "skipped",
            "attempts": attempts,
            "stall_timeout_seconds": args.stall_timeout_seconds,
        },
    )
    final_code = 1
    try:
        for attempt in attempts:
            cmd = _make_train_cmd(args, out, attempt["batch_size"], attempt["grad_accum"])
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "train_start", "attempt": attempt, "cmd": cmd})
            final_code = stream_command(cmd, Path.cwd(), out / "foreground_stdout.log", args.stall_timeout_seconds)
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "train_exit", "attempt": attempt, "code": final_code})
            if final_code == 0:
                break
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "fallback_next", "previous_code": final_code})
        raise SystemExit(final_code)
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def main(argv=None):
    try:
        return _main(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        try:
            parsed, _ = build_parser().parse_known_args(argv)
            out = Path(parsed.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            tb = traceback.format_exc()
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "supervisor_exception", "message": str(exc), "traceback": tb})
            (out / "supervisor_error.txt").write_text(tb, encoding="utf-8")
        except BaseException:
            pass
        raise
    except BaseException as exc:
        try:
            parsed, _ = build_parser().parse_known_args(argv)
            out = Path(parsed.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            tb = traceback.format_exc()
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "supervisor_exception", "message": str(exc), "traceback": tb})
            (out / "supervisor_error.txt").write_text(tb, encoding="utf-8")
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
