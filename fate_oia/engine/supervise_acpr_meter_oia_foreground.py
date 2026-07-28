from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


FALLBACK_LADDER = ((16, 2), (12, 3), (8, 4), (6, 5))


def run_foreground(command: Sequence[str], *, cwd: str | Path, fallback: Sequence[tuple[int, int]] = FALLBACK_LADDER) -> int:
    """Run synchronously and retry only on a diagnosed CUDA OOM.

    No detached process, job, scheduler, or hidden window is used;
    stdout/stderr remain attached to this supervisor's foreground console.
    """
    current = list(command)
    attempts = list(fallback)
    for index, (batch, accumulation) in enumerate(attempts):
        replaced = []
        skip_batch = False
        skip_accum = False
        for token in current:
            if skip_batch:
                skip_batch = False
                continue
            if skip_accum:
                skip_accum = False
                continue
            if token == "--batch_size":
                replaced.extend([token, str(batch)])
                skip_batch = True
            elif token == "--gradient_accumulation_steps":
                replaced.extend([token, str(accumulation)])
                skip_accum = True
            else:
                replaced.append(token)
        if "--batch_size" not in replaced:
            replaced.extend(["--batch_size", str(batch)])
        if "--gradient_accumulation_steps" not in replaced:
            replaced.extend(["--gradient_accumulation_steps", str(accumulation)])
        completed = subprocess.run(replaced, cwd=str(cwd), check=False, env=os.environ.copy())
        if completed.returncode == 0:
            return 0
        if index == len(attempts) - 1:
            return completed.returncode
        # The trainer emits METER_OOM_RETRY only for a recoverable OOM.
        if completed.returncode != 86:
            return completed.returncode
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path.cwd()
    readiness_name = (
        "METER_OIA_V1_PRE_PILOT_READY.json"
        if args.mode == "pilot"
        else "METER_OIA_V1_FULL_TRAIN_READY.json"
    )
    readiness_path = root / ".review" / readiness_name
    if not readiness_path.exists():
        raise SystemExit(f"{readiness_name} is required before {args.mode}")
    command = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_meter_oia",
        "--config", args.config, "--output_dir", args.output_dir,
        "--device", args.device, "--require_ready", "--worktree_root", str(root),
    ]
    if args.mode == "pilot":
        command.extend(["--epochs", "3", "--max_train_samples", "4096", "--max_audit_samples", "1024", "--max_calib_samples", "512", "--max_test_samples", "512"])
    else:
        command.extend(["--epochs", "12"])
    raise SystemExit(run_foreground(command, cwd=root))


if __name__ == "__main__":
    main()
