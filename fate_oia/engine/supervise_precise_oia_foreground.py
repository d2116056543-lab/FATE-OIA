from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from fate_oia.engine.audit_precise_oia_implementation import REQUIRED, _tree_sha


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_review_hash(path: Path, config_path: Path, expected_status: str) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_paths = [Path.cwd() / item for item in REQUIRED]
    functional = record.get("functional_checks", {})
    repo_skill = Path.cwd() / ".codex/skills/precise-oia-implementation-audit/SKILL.md"
    user_skill = Path.home() / ".codex/skills/precise-oia-implementation-audit/SKILL.md"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dino_path = Path.cwd() / config["pretrained_weights"]
    action_schema_path = config_path.parent / "precise_action_semantics.yaml"
    if (
        record.get("status") != expected_status
        or record.get("git_head") != head
        or record.get("config_sha256") != _sha(config_path)
        or record.get("source_tree_sha256") != _tree_sha(source_paths)
        or not repo_skill.exists()
        or not user_skill.exists()
        or record.get("skill_sha256") != _sha(repo_skill)
        or not dino_path.is_file()
        or record.get("pretrained_weights_sha256") != _sha(dino_path)
        or record.get("action_schema_sha256") != _sha(action_schema_path)
        or _sha(repo_skill) != _sha(user_skill)
        or not functional
        or not all(bool(value) for value in functional.values())
        or record.get("unresolved")
    ):
        raise SystemExit("PRECISE review gate is stale for the current HEAD/config")
    if expected_status == "FULL_TRAIN_READY":
        pilot_checks = record.get("pilot_checks", {})
        if not pilot_checks or not all(bool(value) for value in pilot_checks.values()):
            raise SystemExit("PRECISE full gate is missing successful scientific pilot checks")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("PRECISE review gate cannot authorize a dirty worktree")


def assert_fresh_run_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"PRECISE run directory is not empty and could mix stale artifacts: {path}")


def selected_runtime_args(path: Path, fallback_batch: int, fallback_accum: int) -> tuple[int, int]:
    if not path.exists():
        return fallback_batch, fallback_accum
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("valid"):
        raise SystemExit("Selected PRECISE runtime profile is not valid")
    return int(record["batch_size"]), int(record["grad_accum"])


def review_is_current(path: Path, config_path: Path, expected_status: str = "PRE_PILOT_ELIGIBLE") -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        source_paths = [Path.cwd() / item for item in REQUIRED]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dino_path = Path.cwd() / config["pretrained_weights"]
        action_schema_path = config_path.parent / "precise_action_semantics.yaml"
        return record.get("status") == expected_status and record.get("git_head") == head and record.get("config_sha256") == _sha(config_path) and record.get("source_tree_sha256") == _tree_sha(source_paths) and dino_path.is_file() and record.get("pretrained_weights_sha256") == _sha(dino_path) and record.get("action_schema_sha256") == _sha(action_schema_path) and not record.get("unresolved")
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def verify_remote_head() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    remotes = subprocess.check_output(["git", "remote"], text=True).split()
    remote = "github" if "github" in remotes else "origin"
    output = subprocess.check_output(["git", "ls-remote", remote, f"refs/heads/{branch}"], text=True).strip()
    remote_head = output.split()[0] if output else ""
    if remote_head != head:
        raise SystemExit(f"Remote branch HEAD mismatch: local={head}, remote={remote_head or 'missing'}")


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
    if args.mode == "pilot" and args.epochs != 3:
        raise SystemExit("PRECISE pilot epochs != 3; the scientific gate requires exactly three epochs")
    root = Path.cwd()
    review = root / ".review" / "PRECISE_OIA_V1_PRE_PILOT_ELIGIBLE.json"
    audit = [sys.executable, "-m", "fate_oia.engine.audit_precise_oia_implementation", "--config", args.config, "--output_dir", ".review/precise_oia_v1", "--device", args.device, "--mode", "preflight", "--write_pre_pilot_eligible"]
    profile = [sys.executable, "-m", "fate_oia.engine.profile_precise_oia", "--config", args.config, "--output_dir", ".review/precise_oia_v1/runtime", "--device", args.device]
    if args.mode == "preflight":
        verify_remote_head()
        _run(profile)
        _run(audit)
        return
    if not review_is_current(review, Path(args.config)):
        verify_remote_head()
        _run(profile)
        _run(audit)
    verify_review_hash(review, Path(args.config), expected_status="PRE_PILOT_ELIGIBLE")
    verify_remote_head()
    if args.mode == "full" and not (root / ".review" / "PRECISE_OIA_V1_FULL_TRAIN_READY.json").exists():
        raise SystemExit("Full PRECISE training is blocked until current-hash FULL_TRAIN_READY exists")
    if args.mode == "full":
        verify_review_hash(root / ".review" / "PRECISE_OIA_V1_FULL_TRAIN_READY.json", Path(args.config), expected_status="FULL_TRAIN_READY")
    run_dir = Path(args.output_dir) / args.mode
    assert_fresh_run_dir(run_dir)
    selected_batch, selected_accum = selected_runtime_args(root / ".review/precise_oia_v1/runtime/selected_runtime_profile.json", args.batch_size, args.gradient_accumulation_steps)
    command = [sys.executable, "-u", "-m", "fate_oia.engine.train_precise_oia", "--config", args.config, "--output_dir", str(run_dir), "--epochs", str(args.epochs), "--batch_size", str(selected_batch), "--gradient_accumulation_steps", str(selected_accum), "--num_workers", str(args.num_workers), "--device", args.device, "--mode", args.mode]
    if args.mode == "pilot":
        command.extend(["--max_test_samples", "512"])
    _run(command)
    if args.mode == "pilot":
        _run([sys.executable, "-m", "fate_oia.engine.run_precise_pcvl", "--validate_dir", str(run_dir / "pcvl")])
        _run([sys.executable, "-m", "fate_oia.engine.audit_precise_oia_implementation", "--config", args.config, "--output_dir", ".review/precise_oia_v1/post_pilot", "--device", args.device, "--mode", "pilot", "--pilot_dir", str(run_dir), "--write_full_train_ready"])


if __name__ == "__main__":
    main()
