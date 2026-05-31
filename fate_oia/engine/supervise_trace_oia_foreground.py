from __future__ import annotations
import argparse, os, subprocess, sys, traceback
from pathlib import Path
from fate_oia.utils.config_io import load_yaml_config
from fate_oia.utils.trace_artifacts import append_jsonl, write_json


def stream_command(cmd: list[str], cwd: Path, log_path: Path | None = None) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
        return proc.wait()
    finally:
        if log_file is not None:
            log_file.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/fate_oia_train_360x640_trace_oia_v1.yaml"); ap.add_argument("--output_dir", required=True); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch_size", type=int, default=8); ap.add_argument("--grad_accum", type=int, default=4); ap.add_argument("--fallback_batch_size", type=int, default=4); ap.add_argument("--fallback_grad_accum", type=int, default=8); ap.add_argument("--cache_batch_size", type=int, default=2); ap.add_argument("--cache_log_every", type=int, default=1); ap.add_argument("--device", default="cuda"); ap.add_argument("--require_review_pass", action="store_true"); ap.add_argument("--review_pass_path", default=".background_runs/trace_oia_v1_preflight_final_head/REVIEW_PASS_TRACE_OIA.txt"); ap.add_argument("--cache_dir", default=""); ap.add_argument("--skip_cache_build", action="store_true"); ap.add_argument("--disable_feature_cache", action="store_true"); ap.add_argument("--max_train_samples", type=int, default=0); ap.add_argument("--max_test_samples", type=int, default=0); return ap


def _main(argv=None):
    args = build_parser().parse_args(argv)
    pre = Path(args.review_pass_path)
    if args.require_review_pass and not pre.exists():
        raise SystemExit("Missing REVIEW_PASS_TRACE_OIA.txt; refusing training.")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    lock_path = Path(".background_runs/trace_oia_v1_active.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, f"pid={os.getpid()} output_dir={out}\n".encode("utf-8"))
    except FileExistsError as exc:
        raise SystemExit(f"Existing TRACE-OIA lock detected at {lock_path}; remove it only after confirming no TRACE run is active.") from exc
    cfg = load_yaml_config(args.config)
    cache_cfg = cfg.get("feature_cache", {}) if isinstance(cfg, dict) else {}
    cache_required = bool(cache_cfg.get("build_before_training", True)) and not args.disable_feature_cache
    required_hit_rate = float(cache_cfg.get("required_hit_rate", 0.99))
    cache_dir = args.cache_dir or str(out / "dino_token_cache")
    write_json(out / "supervisor_manifest.json", {"foreground": True, "require_review_pass": args.require_review_pass, "review_pass_path": str(pre), "best_selection_split": "test", "epochs": args.epochs, "feature_cache_enabled": not args.disable_feature_cache, "cache_build_before_training": cache_required and not args.skip_cache_build, "cache_dir": cache_dir, "cache_batch_size": args.cache_batch_size, "cache_log_every": args.cache_log_every, "train_batch_size": args.batch_size})
    try:
        if cache_required and not args.skip_cache_build:
            cache_cmd = [sys.executable, "-u", "-m", "fate_oia.engine.build_trace_oia_token_cache", "--config", args.config, "--output_dir", str(out), "--cache_dir", cache_dir, "--batch_size", str(args.cache_batch_size), "--device", args.device, "--required_hit_rate", str(required_hit_rate), "--max_train_samples", str(args.max_train_samples), "--max_test_samples", str(args.max_test_samples), "--log_every", str(args.cache_log_every)]
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "cache_build_start", "cmd": cache_cmd})
            cache_code = stream_command(cache_cmd, Path.cwd(), out / "foreground_stdout.log")
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "cache_build_exit", "code": cache_code})
            if cache_code != 0:
                raise SystemExit(cache_code)
        cmd = [sys.executable, "-u", "-m", "fate_oia.engine.train_trace_oia", "--config", args.config, "--output_dir", str(out), "--cache_dir", cache_dir, "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--gradient_accumulation_steps", str(args.grad_accum), "--device", args.device, "--max_train_samples", str(args.max_train_samples), "--max_test_samples", str(args.max_test_samples)]
        if args.disable_feature_cache:
            cmd.append("--no-feature_cache_enabled")
        code = stream_command(cmd, Path.cwd(), out / "foreground_stdout.log"); append_jsonl(out / "supervisor_decisions.jsonl", {"event": "process_exit", "code": code})
        if code != 0 and args.batch_size != args.fallback_batch_size:
            cmd = [sys.executable, "-u", "-m", "fate_oia.engine.train_trace_oia", "--config", args.config, "--output_dir", str(out), "--cache_dir", cache_dir, "--epochs", str(args.epochs), "--batch_size", str(args.fallback_batch_size), "--gradient_accumulation_steps", str(args.fallback_grad_accum), "--device", args.device, "--max_train_samples", str(args.max_train_samples), "--max_test_samples", str(args.max_test_samples)]
            if args.disable_feature_cache:
                cmd.append("--no-feature_cache_enabled")
            code = stream_command(cmd, Path.cwd(), out / "foreground_stdout.log"); append_jsonl(out / "supervisor_decisions.jsonl", {"event": "fallback_exit", "code": code})
        raise SystemExit(code)
    finally:
        if lock_fd is not None:
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
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "supervisor_exception", "type": type(exc).__name__, "message": str(exc), "traceback": tb})
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
            append_jsonl(out / "supervisor_decisions.jsonl", {"event": "supervisor_exception", "type": type(exc).__name__, "message": str(exc), "traceback": tb})
            (out / "supervisor_error.txt").write_text(tb, encoding="utf-8")
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
