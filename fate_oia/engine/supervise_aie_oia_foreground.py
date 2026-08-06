from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

from fate_oia.utils.aie_hashes import aie_source_tree_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--run-kind", choices=("pilot", "full"), required=True); parser.add_argument("--epochs", type=int, required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int); parser.add_argument("--max-audit-samples", type=int); parser.add_argument("--max-calib-samples", type=int); parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args(); preflight = Path(".background_runs/aie_oia_v1_preflight")
    review_path = preflight / "AIE_IMPLEMENTATION_REVIEW.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config_hash = file_sha256(args.config)
    if not review.get("pass") or review.get("git_head") != current_head or review.get("config_hash") != config_hash:
        raise RuntimeError("AIE implementation REVIEW_PASS is required")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise RuntimeError("AIE full/pilot supervisor requires a clean worktree")
    runtime_path = preflight / "AIE_RUNTIME_PROFILE.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    if not runtime.get("pass") or runtime.get("git_head") != current_head or runtime.get("config_hash") != config_hash:
        raise RuntimeError("AIE runtime profile is required")
    if args.run_kind == "full":
        pilot_dirs = sorted(Path(".background_runs").glob("aie_oia_v1_pilot_*"))
        if not pilot_dirs or not (pilot_dirs[-1] / "AIE_FULL_TRAIN_READY.json").exists():
            raise RuntimeError("AIE PILOT_PASS bound to current implementation is required")
        ready = json.loads((pilot_dirs[-1] / "AIE_FULL_TRAIN_READY.json").read_text(encoding="utf-8"))
        if not ready.get("pass") or ready.get("git_head") != current_head or ready.get("config_hash") != config_hash or ready.get("source_tree_hash") != aie_source_tree_sha256():
            raise RuntimeError("AIE pilot binding does not match current HEAD/config")
    selected = runtime["selected"]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(review_path, output / "AIE_IMPLEMENTATION_REVIEW.json")
    shutil.copy2(runtime_path, output / "AIE_RUNTIME_PROFILE.json")
    command = [
        sys.executable, "-u", "-m", "fate_oia.engine.train_aie_oia", "--config", args.config,
        "--output-dir", args.output_dir, "--run-kind", args.run_kind, "--epochs", str(args.epochs), "--device", args.device,
        "--batch-size", str(selected["batch_size"]), "--gradient-accumulation-steps", str(selected["gradient_accumulation_steps"]),
        "--num-workers", str(selected.get("num_workers", 8)),
    ]
    for name in ("max_train_samples", "max_audit_samples", "max_calib_samples", "max_test_samples"):
        value = getattr(args, name)
        if value is not None:
            command.extend((f"--{name.replace('_', '-')}", str(value)))
    # subprocess.call keeps this supervisor attached and streams child output; it does not daemonize.
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__": main()
