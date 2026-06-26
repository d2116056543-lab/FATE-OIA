from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference_checkpoint", default="")
    parser.add_argument("--require_review_pass", action="store_true")
    args = parser.parse_args()
    if args.require_review_pass and not Path(".background_runs/acpr_vista_v1_preflight/REVIEW_PASS_ACPR_VISTA_V1.txt").exists():
        raise SystemExit("missing REVIEW_PASS_ACPR_VISTA_V1.txt")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_oia",
        "--config",
        "configs/fate_oia_train_360x640_acpr_vista_v1.yaml",
        "--output_dir",
        ".background_runs/acpr_vista_v1_full",
        "--epochs",
        str(args.epochs),
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--persistent_workers",
        "--device",
        args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    raise SystemExit(proc.wait())


if __name__ == "__main__":
    main()
