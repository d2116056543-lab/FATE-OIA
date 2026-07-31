from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fate_oia.utils.meter_artifacts import validate_heca_pilot_bundle


FALLBACK_LADDER = ((6, 5), (5, 6), (4, 8), (3, 10), (2, 15))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_training_readiness(
    review_path: str | Path, pilot_path: str | Path, gate_c_path: str | Path
) -> dict[str, object]:
    head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("HECA full training requires a clean tracked worktree")
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    pilot = json.loads(Path(pilot_path).read_text(encoding="utf-8"))
    gate_c = json.loads(Path(gate_c_path).read_text(encoding="utf-8"))
    if review.get("pass") is not True or review.get("git_head") != head:
        raise RuntimeError("HECA implementation review is missing or stale")
    if pilot.get("pass") is not True or pilot.get("git_head") != head:
        raise RuntimeError("HECA pilot result is missing or stale")
    artifact_failures = validate_heca_pilot_bundle(
        Path(pilot_path).parent, expected_git_head=head
    )
    if artifact_failures:
        raise RuntimeError(
            "HECA pilot artifacts failed strict validation: "
            + ", ".join(artifact_failures)
        )
    gates = pilot.get("gates", {})
    missing = [letter for letter in "ABCDEFG" if gates.get(letter) is not True]
    if missing:
        raise RuntimeError(f"HECA pilot gates are incomplete: {missing}")
    if gate_c.get("pass") is not True or gate_c.get("gate") not in {"C", "HECA_GATE_C"}:
        raise RuntimeError("HECA Gate C artifact is invalid")
    if Path(gate_c_path).resolve().parent != Path(pilot_path).resolve().parent:
        raise RuntimeError("HECA Gate C must come from the signed pilot directory")
    if pilot.get("gate_payloads", {}).get("C") != gate_c:
        raise RuntimeError("HECA Gate C does not match the signed pilot evidence")
    return {"git_head": head, "review": review, "pilot": pilot, "gate_c": gate_c}


def _command(args: argparse.Namespace, batch: int, accum: int, resume: str) -> list[str]:
    command = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_meter_oia",
        "--config", args.config, "--output_dir", args.output_dir,
        "--device", args.device, "--epochs", "14", "--batch_size", str(batch),
        "--gradient_accumulation_steps", str(accum), "--num_workers", str(args.num_workers),
        "--run_kind", "full", "--gate_c_pass", args.gate_c_pass,
        "--test_only", "--no_feature_cache", "--require_no_token_compression",
    ]
    if resume:
        command.extend(("--resume", resume))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--review_pass", required=True)
    parser.add_argument("--pilot_pass", required=True)
    parser.add_argument("--gate_c_pass", required=True)
    args = parser.parse_args()
    validate_training_readiness(args.review_pass, args.pilot_pass, args.gate_c_pass)
    requested = (args.batch_size, args.gradient_accumulation_steps)
    ladder = [requested, *(item for item in FALLBACK_LADDER if item != requested)]
    resume = ""
    for batch, accum in ladder:
        completed = subprocess.run(_command(args, batch, accum, resume), check=False)
        if completed.returncode == 0:
            return
        latest = Path(args.output_dir) / "checkpoint_latest.pth"
        if completed.returncode not in (137, -1073740791) or not latest.exists():
            raise SystemExit(completed.returncode)
        resume = str(latest)
    raise RuntimeError("HECA exhausted the foreground OOM fallback ladder")


if __name__ == "__main__":
    main()
