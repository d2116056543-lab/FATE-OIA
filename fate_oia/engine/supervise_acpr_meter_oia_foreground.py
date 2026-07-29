from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


FALLBACK_LADDER = ((6, 5), (5, 6), (4, 8), (3, 10), (2, 15))
PILOT_GATES = tuple("ABCDEFGH")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_training_readiness(
    review_path: str | Path, pilot_path: str | Path
) -> dict[str, object]:
    review_file = Path(review_path)
    pilot_file = Path(pilot_path)
    if not review_file.exists():
        raise FileNotFoundError(
            f"implementation review pass is required before full training: {review_file}"
        )
    if not pilot_file.exists():
        raise FileNotFoundError(
            f"TESA pilot gate pass is required before full training: {pilot_file}"
        )
    head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Full training requires a clean tracked worktree")
    review = json.loads(review_file.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    if review.get("git_head") != head:
        raise RuntimeError("Implementation review pass does not match current git HEAD")
    if pilot.get("git_head") != head or pilot.get("pass") is not True:
        raise RuntimeError("TESA pilot pass does not match current git HEAD/pass=true")
    gates = pilot.get("gates", {})
    missing = [gate for gate in PILOT_GATES if gates.get(gate) is not True]
    if missing:
        raise RuntimeError(f"TESA pilot gates are incomplete: {missing}")
    return {"git_head": head, "review": review, "pilot": pilot}


def _command(
    args: argparse.Namespace, batch_size: int, grad_accum: int, resume: str
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_meter_oia",
        "--config",
        args.config,
        "--output_dir",
        args.output_dir,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--num_workers",
        str(args.num_workers),
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ]
    if resume:
        command.extend(["--resume", resume])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--review_pass", required=True)
    parser.add_argument("--pilot_pass", required=True)
    args = parser.parse_args()
    validate_training_readiness(args.review_pass, args.pilot_pass)
    requested = (args.batch_size, args.gradient_accumulation_steps)
    ladder = [requested] + [item for item in FALLBACK_LADDER if item != requested]
    resume = ""
    for batch_size, grad_accum in ladder:
        command = _command(args, batch_size, grad_accum, resume)
        print(
            f"tesa_supervisor batch={batch_size} accum={grad_accum} "
            f"resume={resume or 'none'}",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return
        latest = Path(args.output_dir) / "checkpoint_latest.pth"
        if completed.returncode not in (137, -1073740791) or not latest.exists():
            raise SystemExit(completed.returncode)
        resume = str(latest)
    raise RuntimeError("TESA exhausted the foreground OOM fallback ladder")


if __name__ == "__main__":
    main()
