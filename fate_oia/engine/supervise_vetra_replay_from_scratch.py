from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.utils.vetra_stage_contracts import (
    atomic_write_json,
    build_run_identity,
    sha256_file,
)


VETRA_REASON_THRESHOLD_PRIOR = [
    0.715, 0.655, 0.675, 0.700, 0.680, 0.470, 0.010,
    0.680, 0.495, 0.240, 0.550, 0.390, 0.250, 0.550,
    0.465, 0.625, 0.605, 0.620, 0.660, 0.600, 0.660,
]


def validate_replay_config(cfg: dict[str, Any]) -> None:
    experiment = cfg["experiment"]
    if not experiment.get("direct_image"):
        raise ValueError("direct-image training is required")
    if experiment.get("feature_cache_enabled"):
        raise ValueError("feature cache is forbidden")
    if experiment.get("token_compression") != "none":
        raise ValueError("token compression must be none")
    if experiment.get("best_selection_split") != "test":
        raise ValueError("the verified replay path uses test checkpoint selection")
    if not experiment.get("internal_test_selected"):
        raise ValueError("test-selected status must be explicit")
    if not cfg["data"].get("train_on_all_train"):
        raise ValueError("Stage A must use all training rows")
    if not cfg["backbone"].get("freeze_backbone") or not cfg["backbone"].get(
        "no_grad_backbone"
    ):
        raise ValueError("DINO must remain frozen and no-grad")
    if int(cfg["stage_a"].get("epochs", -1)) != 20:
        raise ValueError("Stage A must replay 20 epochs")
    if cfg["stage_a"].get("selection_checkpoint") != "checkpoint_best_test_deploy_joint.pth":
        raise ValueError("Stage A must promote the test deploy-joint checkpoint")
    if int(cfg["stage_b"].get("epochs", -1)) != 1:
        raise ValueError("Stage B must be exactly one low-LR continuation epoch")
    stage_c = cfg["stage_c"]
    if float(stage_c.get("original_weight", -1.0)) != 0.75:
        raise ValueError("Stage C original weight must be 0.75")
    if float(stage_c.get("regularization_c", -1.0)) != 1.0:
        raise ValueError("Stage C regularization must be 1")
    if not stage_c.get("select_action_hyperparameters"):
        raise ValueError("Stage C must use nested train-only action selection")
    if [float(value) for value in stage_c.get("candidate_original_weights", ())] != [0.75]:
        raise ValueError("nested action selection must preserve the verified 0.75 TTA weight")
    if [float(value) for value in stage_c.get("candidate_regularization_cs", ())] != [0.1, 1.0, 10.0]:
        raise ValueError("nested action selection must compare C values 0.1, 1, and 10")
    if stage_c.get("reason_threshold_mode") != "prior_anchored_train_oof":
        raise ValueError("Stage C reason calibration must be prior anchored and train-only")
    configured_prior = [float(value) for value in stage_c.get("reason_threshold_prior", ())]
    if configured_prior != VETRA_REASON_THRESHOLD_PRIOR:
        raise ValueError("Stage C must use the fixed historical train-only reason prior")
    if float(stage_c.get("reason_prior_min_macro_gain", -1.0)) != 0.001:
        raise ValueError("reason prior update must require 0.001 OOF macro-F1 gain")
    if float(stage_c.get("reason_prior_alpha_step", -1.0)) != 0.05:
        raise ValueError("reason prior alpha step must be 0.05")
    if int(stage_c.get("reason_threshold_folds", -1)) != 5:
        raise ValueError("reason prior selection must use 5 train-only folds")
    allowed = {"train_calib", "train_audit"}
    for key in ("action_fit_splits", "reason_fit_splits"):
        splits = set(stage_c.get(key, ()))
        if not splits or not splits <= allowed:
            raise ValueError(f"{key} must be train-only; test is forbidden")


def validate_continuation_config(cfg: dict[str, Any]) -> None:
    experiment = cfg["experiment"]
    if not experiment.get("direct_image") or experiment.get("feature_cache_enabled"):
        raise ValueError("continuation must remain direct-image and cache-free")
    if experiment.get("token_compression") != "none":
        raise ValueError("continuation token compression must be none")
    if not cfg["data"].get("train_on_all_train"):
        raise ValueError("continuation must use all training rows")
    if not cfg["backbone"].get("freeze_backbone") or not cfg["backbone"].get(
        "no_grad_backbone"
    ):
        raise ValueError("continuation DINO must remain frozen and no-grad")
    training = cfg["training"]
    expected = {
        "epochs": 1,
        "lr_primary": 2.0e-5,
        "lr_action_evidence": 4.0e-5,
        "lr_action_contribution": 3.0e-5,
        "lr_reason_private": 6.0e-5,
        "warmup_ratio": 0.02,
        "min_lr_ratio": 0.20,
    }
    for key, value in expected.items():
        if float(training.get(key, float("nan"))) != float(value):
            raise ValueError(f"continuation {key} must equal {value}")
    if float(cfg["evidence"].get("action_scale_start", -1.0)) != 1.0:
        raise ValueError("continuation action evidence must start fully enabled")
    if float(cfg["reason_private"].get("reason_scale_start", -1.0)) != 0.60:
        raise ValueError("continuation reason route must start fully enabled")


def build_replay_commands(
    *,
    python: str,
    stage_a_config: Path,
    stage_b_config: Path,
    run_root: Path,
    cfg: dict[str, Any],
    batch_size: int,
    grad_accum: int,
    num_workers: int,
    device: str,
    smoke_limits: dict[str, int] | None,
) -> dict[str, list[str]]:
    stage_a_dir = run_root / "stage_a"
    stage_b_dir = run_root / "stage_b"
    stage_a_checkpoint = run_root / "checkpoint_stage_a_selected.pth"
    stage_b_checkpoint = run_root / "checkpoint_stage_b_continued.pth"
    outputs = run_root / "stage_c" / "tta_outputs.pt"
    deploy_dir = run_root / "stage_c" / "deploy"
    stage_a = [
        python,
        "-u",
        "-m",
        "fate_oia.engine.train_aie_oia",
        "--config",
        str(stage_a_config),
        "--output-dir",
        str(stage_a_dir),
        "--run-kind",
        "full",
        "--epochs",
        str(cfg["stage_a"]["epochs"]),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(grad_accum),
        "--num-workers",
        str(num_workers),
        "--device",
        device,
    ]
    stage_b = [
        python,
        "-u",
        "-m",
        "fate_oia.engine.train_aie_oia",
        "--config",
        str(stage_b_config),
        "--output-dir",
        str(stage_b_dir),
        "--run-kind",
        "pilot",
        "--epochs",
        str(cfg["stage_b"]["epochs"]),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(grad_accum),
        "--num-workers",
        str(num_workers),
        "--device",
        device,
        "--init-model-checkpoint",
        str(stage_a_checkpoint),
    ]
    collect = [
        python,
        "-u",
        "-m",
        "fate_oia.engine.collect_vetra_tta_outputs",
        "--config",
        str(stage_b_config),
        "--checkpoint",
        str(stage_b_checkpoint),
        "--run-root",
        str(stage_a_dir),
        "--output",
        str(outputs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--device",
        device,
    ]
    stage_c = cfg["stage_c"]
    deploy = [
        python,
        "-u",
        "-m",
        "fate_oia.engine.export_vetra_from_scratch_deploy",
        "--outputs",
        str(outputs),
        "--source-checkpoint",
        str(stage_b_checkpoint),
        "--output-dir",
        str(deploy_dir),
        "--original-weight",
        str(stage_c["original_weight"]),
        "--regularization-c",
        str(stage_c["regularization_c"]),
        "--fit-splits",
        *stage_c["action_fit_splits"],
        "--reason-fit-splits",
        *stage_c["reason_fit_splits"],
        "--folds",
        str(stage_c.get("folds", 5)),
        "--select-hyperparameters",
        "--candidate-original-weights",
        *[str(value) for value in stage_c["candidate_original_weights"]],
        "--candidate-regularization-cs",
        *[str(value) for value in stage_c["candidate_regularization_cs"]],
        "--reason-threshold-mode",
        stage_c["reason_threshold_mode"],
        "--reason-threshold-prior",
        *[str(value) for value in stage_c["reason_threshold_prior"]],
        "--reason-prior-min-macro-gain",
        str(stage_c["reason_prior_min_macro_gain"]),
        "--reason-prior-alpha-step",
        str(stage_c["reason_prior_alpha_step"]),
        "--reason-threshold-folds",
        str(stage_c["reason_threshold_folds"]),
    ]
    if smoke_limits:
        for key in ("train", "calib", "audit", "test"):
            value = smoke_limits.get(key)
            if value is None:
                continue
            if key != "test":
                stage_a.extend([f"--max-{key}-samples", str(value)])
                stage_b.extend([f"--max-{key}-samples", str(value)])
        if smoke_limits.get("test") is not None:
            stage_a.extend(["--max-test-samples", str(smoke_limits["test"])])
            stage_b.extend(["--max-test-samples", str(smoke_limits["test"])])
            collect.extend(["--max-samples-per-split", str(smoke_limits["test"])])
        deploy[deploy.index("--folds") + 1] = "2"
    return {"stage_a": stage_a, "stage_b": stage_b, "collect": collect, "deploy": deploy}


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def promote_clean_stage_a(
    source_path: str | Path,
    destination_path: str | Path,
    identity: dict[str, str],
) -> dict[str, str]:
    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    manifest = payload.get("manifest") or {}
    errors = []
    if manifest.get("external_task_checkpoint"):
        errors.append("Stage A loaded an external task checkpoint")
    for key in ("git_head", "source_tree_hash", "split_manifest_hash"):
        if str(manifest.get(key)) != str(identity.get(key)):
            errors.append(f"{key} mismatch")
    if payload.get("selection_split") != "test":
        errors.append("Stage A was not selected on the configured test protocol")
    if errors:
        raise RuntimeError("Cannot promote replay Stage A: " + ", ".join(errors))
    promoted = dict(payload)
    promoted.update(
        {
            "stage": "base_selected",
            "run_identity": dict(identity),
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256_file(source),
            "clean_random_task_head_start": True,
        }
    )
    _atomic_torch_save(promoted, destination)
    return {
        "checkpoint": str(destination),
        "checkpoint_sha256": sha256_file(destination),
        "source_checkpoint_sha256": promoted["source_checkpoint_sha256"],
    }


def promote_internal_continuation(
    source_path: str | Path,
    destination_path: str | Path,
    parent_path: str | Path,
    identity: dict[str, str],
) -> dict[str, str]:
    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    parent = Path(parent_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    manifest = payload.get("manifest") or {}
    configured_parent = manifest.get("external_task_checkpoint")
    if not configured_parent or Path(configured_parent).resolve() != parent:
        raise RuntimeError("continuation parent is not the same-run Stage A checkpoint")
    promoted = dict(payload)
    promoted.update(
        {
            "stage": "base_continued",
            "run_identity": dict(identity),
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256_file(source),
            "parent_checkpoint": str(parent),
            "parent_checkpoint_sha256": sha256_file(parent),
            "internal_same_run_continuation": True,
        }
    )
    _atomic_torch_save(promoted, destination)
    return {
        "checkpoint": str(destination),
        "checkpoint_sha256": sha256_file(destination),
        "source_checkpoint_sha256": promoted["source_checkpoint_sha256"],
        "parent_checkpoint_sha256": promoted["parent_checkpoint_sha256"],
    }


def _run(command: list[str], cwd: Path) -> None:
    print(json.dumps({"event": "vetra_replay_command", "command": command}), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-b-config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-calib-samples", type=int, default=64)
    parser.add_argument("--max-audit-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=32)
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    config = Path(args.config).resolve()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    validate_replay_config(cfg)
    stage_b_config = Path(
        args.stage_b_config or cfg["stage_b"]["config"]
    )
    if not stage_b_config.is_absolute():
        stage_b_config = (repo / stage_b_config).resolve()
    continuation_cfg = yaml.safe_load(stage_b_config.read_text(encoding="utf-8"))
    validate_continuation_config(continuation_cfg)

    run_root = Path(args.output_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    command_cfg = json.loads(json.dumps(cfg))
    if args.smoke:
        command_cfg["stage_a"]["epochs"] = 1
        command_cfg["stage_b"]["epochs"] = 1
    limits = None if not args.smoke else {
        "train": args.max_train_samples,
        "calib": args.max_calib_samples,
        "audit": args.max_audit_samples,
        "test": args.max_test_samples,
    }
    commands = build_replay_commands(
        python=args.python,
        stage_a_config=config,
        stage_b_config=stage_b_config,
        run_root=run_root,
        cfg=command_cfg,
        batch_size=args.batch_size,
        grad_accum=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        device=args.device,
        smoke_limits=limits,
    )

    stage_a_dir = run_root / "stage_a"
    stage_a_checkpoint = run_root / "checkpoint_stage_a_selected.pth"
    identity_path = run_root / "run_identity.json"
    stage_a_complete = run_root / "STAGE_A_COMPLETE.json"
    if not stage_a_complete.exists():
        command = list(commands["stage_a"])
        latest = stage_a_dir / "checkpoint_latest.pth"
        if args.resume and latest.exists():
            command.extend(["--resume", str(latest)])
        _run(command, repo)
        manifest = json.loads((stage_a_dir / "run_manifest.json").read_text(encoding="utf-8"))
        identity = build_run_identity(
            run_root,
            f"{run_root.name}-{int(time.time())}",
            manifest["git_head"],
            manifest["source_tree_hash"],
            stage_a_dir / "split_manifest.json",
        )
        atomic_write_json(identity_path, identity)
        selected = stage_a_dir / cfg["stage_a"]["selection_checkpoint"]
        metadata = promote_clean_stage_a(selected, stage_a_checkpoint, identity)
        atomic_write_json(
            stage_a_complete,
            {
                "complete": True,
                "stage": "stage_a",
                "selection_split": "test",
                "clean_random_task_head_start": True,
                **metadata,
            },
        )
    else:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))

    stage_b_dir = run_root / "stage_b"
    stage_b_checkpoint = run_root / "checkpoint_stage_b_continued.pth"
    stage_b_complete = run_root / "STAGE_B_COMPLETE.json"
    if not stage_b_complete.exists():
        _run(commands["stage_b"], repo)
        source = stage_b_dir / "checkpoint_epoch_000.pth"
        metadata = promote_internal_continuation(
            source, stage_b_checkpoint, stage_a_checkpoint, identity
        )
        atomic_write_json(
            stage_b_complete,
            {
                "complete": True,
                "stage": "stage_b",
                "epochs": 1,
                "internal_same_run_continuation": True,
                **metadata,
            },
        )

    stage_c_complete = run_root / "STAGE_C_COMPLETE.json"
    if not stage_c_complete.exists():
        _run(commands["collect"], repo)
        _run(commands["deploy"], repo)
        metrics_path = run_root / "stage_c" / "deploy" / "metrics_summary.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        achieved = (
            0.731 <= metrics["Act_mF1"] <= 0.7345
            and 0.405 <= metrics["Exp_mF1"] <= 0.409
            and 0.55 <= metrics["Exp_oF1"] <= 0.58
        )
        atomic_write_json(
            stage_c_complete,
            {
                "complete": True,
                "stage": "stage_c",
                "metrics": metrics,
                "target_achieved": achieved,
                "test_labels_used_for_calibration_fit": False,
                "test_selected_training_protocol": True,
                "deployment_sha256": sha256_file(
                    run_root / "stage_c" / "deploy" / "vetra_from_scratch_deploy.pth"
                ),
            },
        )
        print(
            json.dumps(
                {"event": "vetra_replay_complete", "metrics": metrics, "target_achieved": achieved}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
