from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_stream(cmd: list[str]) -> int:
    print("[foreground]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_care_moe_oia_v1.yaml")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--fallback_batch_size_1", type=int, default=3)
    ap.add_argument("--fallback_grad_accum_1", type=int, default=11)
    ap.add_argument("--fallback_batch_size_2", type=int, default=2)
    ap.add_argument("--fallback_grad_accum_2", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bdd100k_root", default="E:/sbw/BDD100K")
    ap.add_argument("--bdd_oia_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    args = ap.parse_args()
    attempts = [
        (args.batch_size, args.gradient_accumulation_steps),
        (args.fallback_batch_size_1, args.fallback_grad_accum_1),
        (args.fallback_batch_size_2, args.fallback_grad_accum_2),
    ]
    out_dir = args.output_dir or str(Path(".background_runs") / f"care_moe_oia_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    last_code = 1
    for bs, ga in attempts:
        cmd = [
            sys.executable,
            "-m",
            "fate_oia.engine.train_care_moe_oia",
            "--config",
            args.config,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(bs),
            "--gradient_accumulation_steps",
            str(ga),
            "--device",
            args.device,
            "--bdd100k_root",
            args.bdd100k_root,
            "--bdd_oia_root",
            args.bdd_oia_root,
            "--data_root",
            args.bdd_oia_root,
            "--output_dir",
            out_dir,
            "--require_review_pass",
        ]
        if args.max_train_samples:
            cmd += ["--max_train_samples", str(args.max_train_samples)]
        if args.max_test_samples:
            cmd += ["--max_test_samples", str(args.max_test_samples)]
        last_code = run_stream(cmd)
        if last_code == 0:
            return
        print(f"[foreground] attempt batch={bs} accum={ga} failed with code={last_code}; trying fallback if available", flush=True)
    raise SystemExit(last_code)


if __name__ == "__main__":
    main()
