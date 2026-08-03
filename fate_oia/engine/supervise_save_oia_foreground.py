from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fate_oia.engine.evaluate_save_oia_pilot import SAVE_BINDING_KEYS


_BACKGROUND_TOKENS = (
    "start" + "-process",
    "start" + "-job",
    "no" + "hup",
    "--" + "daemon",
    "hid" + "den",
    "scheduled" + " task",
)
_STATIC_BINDING_KEYS = ("git_head", "config_hash", "source_tree_hash", "schema_hash")
_RUNTIME_BINDING_KEYS = ("split_hash", "checkpoint_hash", "logits_hash", "labels_hash", "file_order_hash")


def assert_foreground_command(command: Sequence[str]) -> None:
    rendered = " ".join(map(str, command)).lower()
    found = [token for token in _BACKGROUND_TOKENS if token in rendered]
    if found:
        raise ValueError("SAVE supervisor forbids background process mechanisms: " + ", ".join(found))


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid SAVE readiness JSON: {path}")
    return value


def validate_save_full_readiness(
    review_path: str | Path,
    pilot_path: str | Path,
    profile_path: str | Path,
    ready_path: str | Path,
    *,
    expected_git_head: str,
    allow_numeric_candidate: bool = False,
) -> dict[str, Any]:
    payloads = [_read(path) for path in (review_path, pilot_path, profile_path, ready_path)]
    review, pilot, profile, ready = payloads
    if review.get("pass") is not True or profile.get("pass") is not True:
        raise RuntimeError("SAVE implementation audit or runtime profile is not passing")
    pilot_is_strict = pilot.get("pass") is True and ready.get("pass") is True
    pilot_is_numeric = (
        allow_numeric_candidate
        and pilot.get("numeric_candidate_eligible") is True
        and ready.get("numeric_candidate_eligible") is True
    )
    if not pilot_is_strict and not pilot_is_numeric:
        raise RuntimeError("SAVE pilot is neither strict-pass nor an explicitly allowed safe numeric candidate")
    for payload in payloads:
        bindings = payload.get("bindings")
        if not isinstance(bindings, dict) or any(not bindings.get(key) for key in SAVE_BINDING_KEYS):
            raise RuntimeError("SAVE full training gate has incomplete bindings")
        if bindings["git_head"] != expected_git_head:
            raise RuntimeError("SAVE full training gate is stale for current HEAD")
    canonical = payloads[0]["bindings"]
    if any(
        any(payload["bindings"][key] != canonical[key] for key in _STATIC_BINDING_KEYS)
        for payload in payloads[1:]
    ):
        raise RuntimeError("SAVE review/profile/pilot code bindings do not share one audited source")
    if pilot_is_strict and pilot.get("gates") != {letter: True for letter in "ABCDEFG"}:
        raise RuntimeError("SAVE strict pilot A-G gates are incomplete")
    if any(pilot["bindings"][key] != ready["bindings"][key] for key in _RUNTIME_BINDING_KEYS):
        raise RuntimeError("SAVE ready artifact does not bind the exact pilot outputs")
    chosen = payloads[2].get("chosen")
    if not isinstance(chosen, dict):
        raise RuntimeError("SAVE runtime profile did not select a configuration")
    return {
        "bindings": dict(ready["bindings"]),
        "batch_size": int(chosen["batch_size"]),
        "gradient_accumulation_steps": int(chosen["gradient_accumulation_steps"]),
        "selection_mode": "strict_gates" if pilot_is_strict else "safe_numeric_candidate",
    }


def build_full_command(args: argparse.Namespace, readiness: dict[str, Any]) -> list[str]:
    command = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_save_oia",
        "--config", args.config, "--output-dir", args.output_dir,
        "--run-kind", "full", "--epochs", "12", "--device", args.device,
        "--batch-size", str(readiness["batch_size"]),
        "--gradient-accumulation-steps", str(readiness["gradient_accumulation_steps"]),
        "--num-workers", str(args.num_workers),
    ]
    assert_foreground_command(command)
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--review", required=True)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--allow-numeric-candidate", action="store_true")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("SAVE full training requires a clean audited HEAD")
    readiness = validate_save_full_readiness(
        args.review, args.pilot, args.profile, args.ready, expected_git_head=head,
        allow_numeric_candidate=args.allow_numeric_candidate,
    )
    completed = subprocess.run(build_full_command(args, readiness), check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
