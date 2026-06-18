from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FALLBACKS = [(5, 6), (4, 8), (3, 10), (2, 15)]


def run(cmd: list[str], *, log_path: Path | None = None) -> int:
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    captured: list[str] = []
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with (log_path.open("a", encoding="utf-8") if log_path else open(Path("NUL"), "w")) as log:
        for line in proc.stdout or []:
            print(line, end="", flush=True)
            captured.append(line)
            if log_path:
                log.write(line)
                log.flush()
    code = proc.wait()
    if log_path:
        log_path.write_text(log_path.read_text(encoding="utf-8") + f"\n[exit_code] {code}\n", encoding="utf-8")
    return code


def run_required(cmd: list[str], *, log_path: Path | None = None) -> None:
    code = run(cmd, log_path=log_path)
    if code != 0:
        raise SystemExit(code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_seca_v1.yaml")
    ap.add_argument("--output_dir", default=".background_runs/acpr_seca_v1_full")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    log_path = out_dir / "supervisor.log"
    if args.require_review_pass and not Path(".background_runs/acpr_seca_v1_preflight/REVIEW_PASS_ACPR_SECA_V1.txt").exists():
        raise SystemExit("Missing REVIEW_PASS_ACPR_SECA_V1.txt")
    run_required([
        sys.executable, "-m", "fate_oia.engine.audit_acpr_seca_implementation",
        "--config", args.config,
        "--output_dir", ".background_runs/acpr_seca_v1_preflight_runtime",
        "--device", args.device,
        "--write_review_pass",
    ], log_path=log_path)
    run_required([
        sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_oia",
        "--config", args.config,
        "--output_dir", ".background_runs/acpr_seca_v1_supervisor_smoke",
        "--epochs", "1",
        "--batch_size", "1",
        "--gradient_accumulation_steps", "2",
        "--max_train_samples", "8",
        "--max_test_samples", "8",
        "--num_workers", "0",
        "--device", args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ], log_path=log_path)
    ladder = [(args.batch_size, args.gradient_accumulation_steps)]
    for item in FALLBACKS:
        if item not in ladder:
            ladder.append(item)
    for batch_size, accum in ladder:
        cmd = [
            sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_oia",
            "--config", args.config,
            "--output_dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(batch_size),
            "--gradient_accumulation_steps", str(accum),
            "--num_workers", str(args.num_workers),
            "--device", args.device,
            "--test_only",
            "--no_feature_cache",
            "--require_no_token_compression",
        ]
        code = run(cmd, log_path=log_path)
        if code == 0:
            return
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower() if log_path.exists() else ""
        if "out of memory" not in text and "cuda error: out of memory" not in text:
            raise SystemExit(code)
        print(f"[supervisor] OOM detected; fallback from batch={batch_size}, accum={accum}", flush=True)
    raise SystemExit("All SECA OOM fallbacks failed")


if __name__ == "__main__":
    main()
