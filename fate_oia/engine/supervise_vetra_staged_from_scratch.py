from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from fate_oia.utils.vetra_stage_contracts import (
    atomic_write_json,
    build_run_identity,
    promote_stage_a_checkpoint,
    sha256_file,
    validate_stage_checkpoint,
)


def validate_staged_config(cfg: dict) -> None:
    experiment = cfg["experiment"]
    if not experiment.get("direct_image"):
        raise ValueError("direct-image training is required")
    if experiment.get("feature_cache_enabled"):
        raise ValueError("feature cache is forbidden")
    if experiment.get("token_compression") != "none":
        raise ValueError("token compression is forbidden")
    if experiment.get("best_selection_split") != "train_audit":
        raise ValueError("Stage A selection must use train_audit")
    if not cfg["backbone"].get("freeze_backbone") or not cfg["backbone"].get("no_grad_backbone"):
        raise ValueError("DINO must remain frozen and no-grad")
    if cfg["stage_c"].get("reason_fit_splits") != ["train_calib"]:
        raise ValueError("reason thresholds must fit train_calib only")
    for name in ("action_fit_splits", "reason_fit_splits"):
        if "test" in cfg["stage_c"][name]:
            raise ValueError("test cannot fit deployment parameters")


def build_stage_commands(
    *, python: str, config: Path, run_root: Path, cfg: dict,
    batch_size: int, grad_accum: int, num_workers: int, device: str,
    smoke_limits: dict | None,
) -> dict[str, list[str]]:
    stage_a_dir = run_root / "stage_a"
    stage_b_dir = run_root / "stage_b"
    stage_a_checkpoint = run_root / "checkpoint_stage_a_selected.pth"
    stage_b_checkpoint = stage_b_dir / "checkpoint_stage_b_selected.pth"
    identity = run_root / "run_identity.json"
    outputs = run_root / "stage_c" / "tta_outputs.pt"
    deploy_dir = run_root / "stage_c" / "deploy"
    stage_a = [
        python, "-u", "-m", "fate_oia.engine.train_aie_oia",
        "--config", str(config), "--output-dir", str(stage_a_dir),
        "--run-kind", "full", "--epochs", str(cfg["stage_a"]["epochs"]),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(grad_accum),
        "--num-workers", str(num_workers), "--device", device,
    ]
    stage_b = [
        python, "-u", "-m", "fate_oia.engine.train_vetra_staged_refine",
        "--config", str(config), "--stage-a-checkpoint", str(stage_a_checkpoint),
        "--run-identity", str(identity), "--output-dir", str(stage_b_dir),
        "--epochs", str(cfg["stage_b"]["epochs"]),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(grad_accum),
        "--num-workers", str(num_workers), "--device", device,
    ]
    collect = [
        python, "-u", "-m", "fate_oia.engine.collect_vetra_tta_outputs",
        "--config", str(config), "--checkpoint", str(stage_a_checkpoint),
        "--stage-b-checkpoint", str(stage_b_checkpoint),
        "--run-root", str(stage_a_dir), "--output", str(outputs),
        "--batch-size", str(batch_size), "--num-workers", str(num_workers),
        "--device", device,
    ]
    stage_c = cfg["stage_c"]
    deploy = [
        python, "-u", "-m", "fate_oia.engine.export_vetra_from_scratch_deploy",
        "--outputs", str(outputs), "--source-checkpoint", str(stage_b_checkpoint),
        "--output-dir", str(deploy_dir),
        "--fit-splits", *stage_c["action_fit_splits"],
        "--reason-fit-splits", *stage_c["reason_fit_splits"],
        "--select-hyperparameters", "--stable-action-thresholds",
        "--folds", str(stage_c.get("folds", 5)),
        "--selection-outer-folds", str(stage_c.get("selection_outer_folds", 5)),
        "--selection-inner-folds", str(stage_c.get("selection_inner_folds", 4)),
        "--threshold-folds", str(stage_c.get("threshold_folds", 10)),
        "--candidate-original-weights", *map(str, stage_c.get(
            "original_weights", [0.0, 0.25, 0.5, 0.75, 1.0]
        )),
        "--candidate-regularization-cs", *map(str, stage_c.get(
            "regularization_cs", [0.01, 0.1, 1.0, 10.0]
        )),
    ]
    if smoke_limits:
        for key in ("train", "calib", "audit", "test"):
            value = smoke_limits.get(key)
            if value is None:
                continue
            cli = f"--max-{key}-samples"
            stage_a.extend([cli, str(value)])
            if key in ("train", "calib", "audit"):
                stage_b.extend([cli, str(value)])
        maximum = smoke_limits.get("test")
        if maximum is not None:
            collect.extend(["--max-samples-per-split", str(maximum)])
        deploy[deploy.index("--folds") + 1] = "2"
        deploy[deploy.index("--selection-outer-folds") + 1] = "2"
        deploy[deploy.index("--selection-inner-folds") + 1] = "2"
        deploy[deploy.index("--threshold-folds") + 1] = "2"
    return {"stage_a": stage_a, "stage_b": stage_b, "collect": collect, "deploy": deploy}


def _run(command: list[str], cwd: Path) -> None:
    print(json.dumps({"event": "staged_command", "command": command}), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=8)
    parser.add_argument("--max-calib-samples", type=int, default=8)
    parser.add_argument("--max-audit-samples", type=int, default=8)
    parser.add_argument("--max-test-samples", type=int, default=8)
    args = parser.parse_args()

    config = Path(args.config).resolve()
    repo = Path.cwd().resolve()
    run_root = Path(args.output_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    validate_staged_config(cfg)
    if args.smoke:
        cfg = json.loads(json.dumps(cfg))
        cfg["stage_a"]["epochs"] = 1
        cfg["stage_b"]["epochs"] = 1
    limits = None if not args.smoke else {
        "train": args.max_train_samples,
        "calib": args.max_calib_samples,
        "audit": args.max_audit_samples,
        "test": args.max_test_samples,
    }
    commands = build_stage_commands(
        python=args.python, config=config, run_root=run_root, cfg=cfg,
        batch_size=args.batch_size, grad_accum=args.gradient_accumulation_steps,
        num_workers=args.num_workers, device=args.device, smoke_limits=limits,
    )
    stage_a_dir = run_root / "stage_a"
    promoted = run_root / "checkpoint_stage_a_selected.pth"
    identity_path = run_root / "run_identity.json"
    stage_a_complete = run_root / "STAGE_A_COMPLETE.json"

    if not stage_a_complete.exists():
        command = list(commands["stage_a"])
        latest = stage_a_dir / "checkpoint_latest.pth"
        if args.resume and latest.exists():
            command.extend(["--resume", str(latest)])
        _run(command, repo)
        source = stage_a_dir / "checkpoint_final_train_audit_selected.pth"
        manifest = json.loads((stage_a_dir / "run_manifest.json").read_text(encoding="utf-8"))
        run_id = f"{run_root.name}-{int(time.time())}"
        identity = build_run_identity(
            run_root, run_id, manifest["git_head"], manifest["source_tree_hash"],
            stage_a_dir / "split_manifest.json",
        )
        atomic_write_json(identity_path, identity)
        metadata = promote_stage_a_checkpoint(source, promoted, identity)
        atomic_write_json(stage_a_complete, {
            "complete": True, "stage": "stage_a", **metadata,
            "selection_split": "train_audit",
        })
    else:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        validate_stage_checkpoint(promoted, identity, expected_stage="base_selected")

    stage_b_checkpoint = run_root / "stage_b" / "checkpoint_stage_b_selected.pth"
    if not (run_root / "stage_b" / "STAGE_B_COMPLETE.json").exists():
        _run(commands["stage_b"], repo)
    validate_stage_checkpoint(
        stage_b_checkpoint, identity, expected_stage="action_refined",
        expected_parent_sha256=sha256_file(promoted),
    )

    stage_c_complete = run_root / "STAGE_C_COMPLETE.json"
    if not stage_c_complete.exists():
        _run(commands["collect"], repo)
        _run(commands["deploy"], repo)
        metrics_path = run_root / "stage_c" / "deploy" / "metrics_summary.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        achieved = (
            metrics["Act_mF1"] >= 0.731
            and metrics["Exp_mF1"] >= 0.405
            and 0.55 <= metrics["Exp_oF1"] <= 0.57
        )
        atomic_write_json(stage_c_complete, {
            "complete": True, "stage": "stage_c", "metrics": metrics,
            "target_achieved": achieved,
            "test_labels_used_for_parameters": False,
            "deployment_sha256": sha256_file(
                run_root / "stage_c" / "deploy" / "vetra_from_scratch_deploy.pth"
            ),
        })
        print(json.dumps({"event": "staged_complete", "metrics": metrics, "target_achieved": achieved}), flush=True)


if __name__ == "__main__":
    main()
