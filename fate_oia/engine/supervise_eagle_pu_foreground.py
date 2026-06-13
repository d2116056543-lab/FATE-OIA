from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_once(args: argparse.Namespace, batch_size: int, grad_accum: int) -> tuple[int, list[str]]:
    cmd = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_eagle_pu_oia",
        "--config", args.config,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--device", args.device,
        "--test_only", "--no_feature_cache", "--require_no_token_compression",
    ]
    print("foreground_cmd=" + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        tail.append(line)
        tail = tail[-80:]
        print(line, end="", flush=True)
    return proc.wait(), tail


def is_oom(lines: list[str]) -> bool:
    text = "".join(lines).lower()
    return "out of memory" in text or "cuda error: out of memory" in text or "cublas_status_alloc_failed" in text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    if args.require_review_pass and not Path(".background_runs/eagle_pu_v1_preflight/REVIEW_PASS_EAGLE_PU_V1.txt").exists():
        raise SystemExit("Missing REVIEW_PASS_EAGLE_PU_V1.txt")
    ladder = [(args.batch_size, args.gradient_accumulation_steps), (5, 6), (4, 8), (3, 11), (2, 16), (1, 30)]
    seen: set[tuple[int, int]] = set()
    for idx, (batch, accum) in enumerate(ladder):
        if (batch, accum) in seen:
            continue
        seen.add((batch, accum))
        emergency = (batch, accum) == (1, 30)
        print(f"eagle_pu_supervisor_attempt batch_size={batch} grad_accum={accum} emergency_fallback={emergency}", flush=True)
        code, tail = run_once(args, batch, accum)
        if code == 0:
            raise SystemExit(0)
        if is_oom(tail) and idx < len(ladder) - 1:
            print(f"eagle_pu_supervisor_oom_fallback from=({batch},{accum})", flush=True)
            continue
        print("eagle_pu_supervisor_non_oom_failure", flush=True)
        raise SystemExit(code)
    raise SystemExit(1)

if __name__ == "__main__":
    main()
