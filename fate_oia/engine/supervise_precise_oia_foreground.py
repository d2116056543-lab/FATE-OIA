from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "pilot", "full"), default="preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path.cwd()
    review = root / ".review" / "PRECISE_OIA_V1_PRE_PILOT_ELIGIBLE.json"
    audit = [sys.executable, "-m", "fate_oia.engine.audit_precise_oia_implementation", "--config", args.config, "--output_dir", ".review/precise_oia_v1", "--device", args.device, "--mode", "preflight", "--write_pre_pilot_eligible"]
    if args.mode == "preflight":
        _run(audit)
        return
    if not review.exists():
        _run(audit)
    if args.mode == "full" and not (root / ".review" / "PRECISE_OIA_V1_FULL_TRAIN_READY.json").exists():
        raise SystemExit("Full PRECISE training is blocked until current-hash FULL_TRAIN_READY exists")
    run_dir = Path(args.output_dir) / args.mode
    command = [sys.executable, "-u", "-m", "fate_oia.engine.train_precise_oia", "--config", args.config, "--output_dir", str(run_dir), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), "--num_workers", str(args.num_workers), "--device", args.device]
    if args.mode == "pilot":
        command.extend(["--max_train_samples", "4096", "--max_test_samples", "512"])
    _run(command)


if __name__ == "__main__":
    main()
