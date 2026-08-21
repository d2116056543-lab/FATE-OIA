from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from fate_oia.utils.tida_artifacts import atomic_write_json, file_sha256, validate_completion_artifact


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_ready(ready_path: Path, config: Path, manifest: Path, image_checkpoint: Path) -> dict:
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    failures = validate_completion_artifact(ready, phase="full_train_ready")
    expected = {
        "git_head": _git("rev-parse", "HEAD"), "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "config_sha256": file_sha256(config), "clip_manifest_sha256": file_sha256(manifest),
        "image_checkpoint_sha256": file_sha256(image_checkpoint),
    }
    failures.extend(f"{key} mismatch" for key, value in expected.items() if ready.get(key) != value)
    if _git("status", "--porcelain", "--untracked-files=all"):
        failures.append("worktree is not clean")
    if failures:
        raise RuntimeError("FULL_TRAIN_READY is invalid: " + "; ".join(sorted(set(failures))))
    return ready


def run_visible(command: list[str]) -> None:
    print(json.dumps({"event": "tida_supervisor_command", "command": command}, ensure_ascii=False), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--review-ready", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--context-chunk-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config, manifest = Path(args.config), Path(args.clip_manifest)
    image_checkpoint, output_dir = Path(args.image_checkpoint), Path(args.output_dir)
    ready = validate_ready(Path(args.review_ready), config, manifest, image_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "FOREGROUND_SUPERVISOR_STARTED.json", {
        "pass": True, "git_head": ready["git_head"], "review_ready": str(Path(args.review_ready).resolve())
    })
    train_command = [
        args.python, "-u", "-m", "fate_oia.engine.train_tida_oia",
        "--config", str(config), "--clip-manifest", str(manifest),
        "--image-checkpoint", str(image_checkpoint), "--output-dir", str(output_dir),
        "--epochs", "10", "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--context-chunk-size", str(args.context_chunk_size), "--num-workers", str(args.num_workers),
        "--device", args.device, "--run-kind", "full",
    ]
    if args.resume:
        train_command.extend(["--resume", args.resume])
    try:
        run_visible(train_command)
        train_completed = json.loads((output_dir / "TRAIN_COMPLETED_TIDA_OIA_V1.json").read_text(encoding="utf-8"))
        if train_completed.get("pass") is not True or int(train_completed.get("epochs_completed", 0)) != 10:
            raise RuntimeError("trainer exited without a valid ten-epoch completion marker")
        best_checkpoint = output_dir / "checkpoint_best_test_joint.pth"
        tta_dir = output_dir / "stage_c_tta"
        run_visible([
            args.python, "-u", "-m", "fate_oia.engine.collect_tida_tta_outputs",
            "--config", str(config), "--checkpoint", str(best_checkpoint),
            "--clip-manifest", str(manifest), "--image-checkpoint", str(image_checkpoint),
            "--output-dir", str(tta_dir), "--device", args.device,
            "--batch-size", str(args.batch_size), "--context-chunk-size", str(args.context_chunk_size),
            "--num-workers", str(args.num_workers),
        ])
        run_visible([
            args.python, "-u", "-m", "fate_oia.engine.export_tida_deployment",
            "--tta-output-dir", str(tta_dir), "--checkpoint", str(best_checkpoint),
            "--output-dir", str(output_dir / "stage_c_deployment"),
        ])
    except BaseException as error:
        atomic_write_json(output_dir / "FOREGROUND_SUPERVISOR_FAILED.json", {
            "pass": False, "error_type": type(error).__name__, "error": str(error)
        })
        raise
    atomic_write_json(output_dir / "FOREGROUND_TRAIN_COMPLETED.json", {
        "pass": True, "git_head": ready["git_head"], "epochs": 10,
        "stage_c_deployment": str((output_dir / "stage_c_deployment" / "tida_oia_v1_deploy.pth").resolve()),
    })


if __name__ == "__main__":
    main()
