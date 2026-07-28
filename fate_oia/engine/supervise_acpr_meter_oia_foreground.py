from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from fate_oia.engine.train_acpr_meter_oia import validate_training_readiness
from fate_oia.utils.meter_artifacts import python_source_tree_hash


# The real-DINO profile selected 6/5. Larger candidates exceeded the
# reserved-memory limit, so retries may only reduce the physical batch.
FALLBACK_LADDER = ((6, 5), (4, 8), (3, 11), (2, 16))
PILOT_READY_NAME = "METER_OIA_V1_PRE_PILOT_READY.json"
FULL_READY_NAME = "METER_OIA_V1_FULL_TRAIN_READY.json"


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
    epochs = 3 if args.mode == "pilot" else 12
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    clean_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    remote_line = subprocess.check_output(
        ["git", "ls-remote", "github", f"refs/heads/{git_branch}"],
        cwd=root,
        text=True,
    ).strip()
    remote_head = remote_line.split()[0] if remote_line else ""
    validate_training_readiness(
        root=root,
        config_path=(root / args.config).resolve(),
        epochs=epochs,
        use_mock_dino=False,
        git_head=git_head,
        git_branch=git_branch,
        remote_head=remote_head,
        clean_status=clean_status,
        source_tree_hash=python_source_tree_hash(root),
    )
    readiness_name = PILOT_READY_NAME if args.mode == "pilot" else FULL_READY_NAME
    print(f"validated readiness: {readiness_name} HEAD={git_head}", flush=True)
    command = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_meter_oia",
        "--config", args.config, "--output_dir", args.output_dir,
        "--device", args.device, "--require_ready", "--worktree_root", str(root),
    ]
    if args.mode == "pilot":
        command.extend(["--epochs", "3", "--max_train_samples", "4096", "--max_audit_samples", "1024", "--max_calib_samples", "512", "--max_test_samples", "512"])
    else:
        command.extend(["--epochs", str(epochs)])
    raise SystemExit(run_foreground(command, cwd=root))


if __name__ == "__main__":
    main()
