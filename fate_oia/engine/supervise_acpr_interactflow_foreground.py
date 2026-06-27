from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_interactflow_psi",
        "--config",
        args.config,
        "--output_dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "--device",
        args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ]
    print("foreground_exec " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

