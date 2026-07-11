from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex
from fate_oia.datasets.mosaic_grounding_observations import MOSAICGroundingObservationBuilder
from fate_oia.datasets.mosaic_multiview import MOSAICWeakMultiView
from fate_oia.engine.mosaic_schedule import mosaic_phase_controls
from fate_oia.engine.train_acpr_mosaic_ad import (
    _apply_phase,
    build_loaders,
    build_model_components,
    build_optimizers,
    evaluate_factor_modes,
    fit_calibrator,
    load_config,
    train_representation_epoch,
)
from fate_oia.engine.eval_acpr_mosaic_ad import evaluate_mosaic
from fate_oia.optim.mosaic_action_anchor import MOSAICActionAnchoredGradient
from fate_oia.utils.mosaic_artifacts import write_json


def _metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    deploy = evaluation["metrics_summary"]["deploy_fixed"]
    return {
        "Act_mF1": float(deploy["Act_mF1"]),
        "Act_oF1": float(deploy["Act_oF1"]),
        "Act_mAP": float(deploy["Act_mAP"]),
        "Exp_mF1": float(deploy["Exp_mF1"]),
        "Exp_oF1": float(deploy["Exp_oF1"]),
        "Exp_mAP": float(deploy["Exp_mAP"]),
        "joint": float(deploy["joint"]),
    }


def _active_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("available", True)]


def _summarize_training_rows(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    loss_rows = rows["loss_components.jsonl"]
    anchors = _active_rows(rows["action_anchor_stats.jsonl"])
    selective = rows["selective_observation_stats.jsonl"]
    recovery = next(
        (row for row in reversed(rows["posterior_recovery_stats.jsonl"]) if row.get("summary")),
        {},
    )
    cosines = []
    for row in anchors:
        denominator = float(row["action_grad_norm"]) * float(row["aux_grad_norm"])
        if denominator > 0:
            cosines.append(float(row["dot_action_aux"]) / denominator)
    finite_losses = all(
        math.isfinite(float(value))
        for row in loss_rows
        for name, value in row.items()
        if name.startswith("loss_") or name.endswith("_loss")
    )
    return {
        "steps": len(loss_rows),
        "finite_losses": finite_losses,
        "loader_stalls": sum(bool(row.get("dataloader_stall")) for row in loss_rows),
        "posterior_active_rate": sum(bool(row.get("posterior_available")) for row in selective)
        / max(len(selective), 1),
        "posterior_mean": sum(float(row["posterior_mean"]) for row in selective) / max(len(selective), 1),
        "propensity_mean": sum(float(row["propensity_mean"]) for row in selective) / max(len(selective), 1),
        "anchor_pass_rate": sum(bool(row.get("constraint_pass")) for row in anchors) / max(len(anchors), 1),
        "anchor_cosine_mean": sum(cosines) / max(len(cosines), 1),
        "anchor_cosine_min": min(cosines) if cosines else 0.0,
        "posterior_recovery": recovery,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    config = load_config(args.config)
    device = torch.device(args.device)
    train_loader, calib_loader, test_loader, split_stats = build_loaders(
        config,
        output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_calib_samples=args.max_calib_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    model, selective, threshold, action_queue, reason_queue = build_model_components(
        config, args.config, device
    )
    representation_optimizer, calibration_optimizer = build_optimizers(model, selective, threshold, config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if int(checkpoint["epoch"]) != 4:
        raise ValueError("component diagnostic requires the completed epoch-4 checkpoint")
    model.load_state_dict(checkpoint["model"])
    selective.load_state_dict(checkpoint["selective_observation"])
    threshold.load_state_dict(checkpoint["calibrator"])
    action_queue.load_state_dict(checkpoint["action_queue"])
    reason_queue.load_state_dict(checkpoint["reason_queue"])
    representation_optimizer.load_state_dict(checkpoint["optimizer"])

    controls = mosaic_phase_controls(args.phase_epoch)
    if controls.phase != "D_joint_ranking" or not controls.posterior_enabled:
        raise ValueError("component diagnostic must exercise the complete Phase-D path")
    _apply_phase(model, selective, representation_optimizer, controls)
    before = evaluate_mosaic(model, threshold, test_loader, device, epoch=args.phase_epoch)

    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    grounding_builder = MOSAICGroundingObservationBuilder(model.schema_bundle["factors"])
    grounding_index = BDD100KGroundingIndex(config["data"]["bdd100k_root"])
    multiview = MOSAICWeakMultiView(factor_names, seed=args.seed)
    action_anchor = MOSAICActionAnchoredGradient(
        aux_shared_lambda_max=float(config["optimizer"]["aux_shared_lambda_max"]),
        action_anchor_kappa=float(config["optimizer"]["action_anchor_kappa"]),
    )
    rows, global_update = train_representation_epoch(
        model=model,
        selective=selective,
        action_queue=action_queue,
        reason_queue=reason_queue,
        loader=train_loader,
        optimizer=representation_optimizer,
        action_anchor=action_anchor,
        grounding_builder=grounding_builder,
        grounding_index=grounding_index,
        multiview=multiview,
        controls=controls,
        config=config,
        device=device,
        epoch=args.phase_epoch,
        grad_accum=args.grad_accum,
        global_update=0,
        total_updates=max(1, math.ceil(len(train_loader) / args.grad_accum)),
        profile_timing=True,
    )
    threshold_rows = fit_calibrator(
        model,
        threshold,
        calib_loader,
        calibration_optimizer,
        device,
        epoch=args.phase_epoch,
        max_steps=int(config["calibration"]["steps_per_epoch"]),
        calibration_config=config["calibration"],
    )
    after = evaluate_mosaic(model, threshold, test_loader, device, epoch=args.phase_epoch)
    factor_modes = evaluate_factor_modes(
        model,
        calib_loader,
        grounding_builder,
        grounding_index,
        device,
        epoch=args.phase_epoch,
        max_samples=min(args.max_calib_samples, int(config["training"]["factor_audit_samples"])),
    )
    training = _summarize_training_rows(rows)
    before_metrics = _metrics(before)
    after_metrics = _metrics(after)
    action_branch = after["action_branch_metrics"]
    state_gate_effect = abs(
        float(action_branch["raw"]["Act_mF1"]) - float(action_branch["visual"]["Act_mF1"])
    )
    checks = {
        "all_losses_finite": training["finite_losses"],
        "no_loader_stall": training["loader_stalls"] == 0,
        "posterior_fully_active": training["posterior_active_rate"] == 1.0,
        "posterior_not_collapsed": 0.01 < training["posterior_mean"] < 0.99,
        "propensity_not_collapsed": 0.05 < training["propensity_mean"] < 0.95,
        "action_anchor_pass": training["anchor_pass_rate"] >= 0.95,
        "anchor_cosine_finite": math.isfinite(training["anchor_cosine_mean"]),
        "state_action_path_active": state_gate_effect > 1e-6,
        "content_factor_active": factor_modes["full_factor_metric"] > factor_modes["prior_only_factor_metric"],
        "content_only_retained": factor_modes["content_only_retention"] >= 0.70,
        "action_no_short_collapse": after_metrics["Act_mF1"] >= before_metrics["Act_mF1"] - 0.02,
        "reason_map_no_short_collapse": after_metrics["Exp_mAP"] >= before_metrics["Exp_mAP"] - 0.02,
        "threshold_rows_complete": len(threshold_rows) == int(config["calibration"]["steps_per_epoch"]),
    }
    diagnostic_checkpoint = output_dir / "checkpoint_component_diagnostic.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "selective_observation": selective.state_dict(),
            "calibrator": threshold.state_dict(),
            "action_queue": action_queue.state_dict(),
            "reason_queue": reason_queue.state_dict(),
            "optimizer": representation_optimizer.state_dict(),
            "epoch": args.phase_epoch,
            "phase": controls.phase,
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
        },
        diagnostic_checkpoint,
    )
    result = {
        "pass": all(checks.values()),
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "phase_epoch": args.phase_epoch,
        "phase": controls.phase,
        "sample_counts": split_stats,
        "before": before_metrics,
        "after": after_metrics,
        "delta": {name: after_metrics[name] - before_metrics[name] for name in before_metrics},
        "action_branches": action_branch,
        "state_gate_effect_mf1": state_gate_effect,
        "factor_modes": factor_modes,
        "training": training,
        "checks": checks,
        "diagnostic_checkpoint": str(diagnostic_checkpoint.resolve()),
        "global_update": global_update,
    }
    write_json(output_dir / "component_diagnostic.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase_epoch", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=512)
    parser.add_argument("--max_calib_samples", type=int, default=256)
    parser.add_argument("--max_test_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
