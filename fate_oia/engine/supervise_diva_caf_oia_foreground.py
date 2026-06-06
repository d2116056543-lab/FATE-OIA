from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class RequireReviewPass(RuntimeError):
    pass


def _run_stream(cmd: list[str]) -> tuple[int, list[str]]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    lines: list[str] = []
    try:
        for line in proc.stdout:
            lines.append(line)
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    return proc.wait(), lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Foreground supervisor for DIVA-CAF-OIA V2")
    parser.add_argument("--review_pass", default=".background_runs/diva_caf_oia_v2_preflight/REVIEW_PASS_DIVA_CAF_OIA_V2.txt")
    parser.add_argument("--output_dir", default=".background_runs/diva_caf_oia_v2_full")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--fallback_batch_size_1", type=int, default=3)
    parser.add_argument("--fallback_grad_accum_1", type=int, default=11)
    parser.add_argument("--fallback_batch_size_2", type=int, default=2)
    parser.add_argument("--fallback_grad_accum_2", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not Path(args.review_pass).exists():
        raise RequireReviewPass(f"missing review pass: {args.review_pass}")
    attempts = [
        (args.batch_size, args.grad_accum),
        (args.fallback_batch_size_1, args.fallback_grad_accum_1),
        (args.fallback_batch_size_2, args.fallback_grad_accum_2),
    ]
    for idx, (bs, accum) in enumerate(attempts):
        cmd = [
            sys.executable, "-m", "fate_oia.engine.train_diva_caf_oia",
            "--config", "configs/fate_oia_train_360x640_diva_caf_oia_v2.yaml",
            "--output_dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(bs),
            "--gradient_accumulation_steps", str(accum),
            "--device", args.device,
            "--no_feature_cache",
            "--test_only",
            "--require_review_pass",
            "--print_every", "200",
        ]
        print(f"Foreground attempt={idx} batch_size={bs} grad_accum={accum}", flush=True)
        rc, lines = _run_stream(cmd)
        if rc == 0:
            return
        joined = "".join(lines[-80:]).lower()
        if "out of memory" not in joined and "cuda" not in joined:
            raise SystemExit(rc)
        print("CUDA memory failure detected; trying configured Fallback.", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
