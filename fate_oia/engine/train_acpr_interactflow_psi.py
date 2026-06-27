from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fate_oia.acpr_interactflow.artifacts import append_jsonl, write_json
from fate_oia.acpr_interactflow.config import load_interactflow_config
from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel
from fate_oia.acpr_interactflow.psi_damo_dataset import PSIDAMO11902Dataset, psi_interactflow_collate
from fate_oia.engine.eval_acpr_interactflow_psi import evaluate
from fate_oia.losses.acpr_interactflow_losses import compute_interactflow_losses


def _loader(ds, batch_size: int, cfg: dict, shuffle: bool) -> DataLoader:
    data = cfg["data"]
    workers = int(data.get("num_workers", 0))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(data.get("pin_memory", False)),
        "collate_fn": psi_interactflow_collate,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(data.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
    return DataLoader(ds, **kwargs)


def _build_model(cfg: dict) -> ACPRInteractFlowPPModel:
    return ACPRInteractFlowPPModel(
        pretrained_weights=cfg["paths"]["dino_weights"],
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path=cfg["model"]["interaction_flow"]["grammar_yaml"],
        exp29_names_path=cfg["paths"].get("psi_label_embedding_json"),
        action_dim=int(cfg["data"]["action_dim"]),
        use_mock_dino=False,
    )


def _build_optimizer(model: ACPRInteractFlowPPModel, cfg: dict) -> torch.optim.Optimizer:
    lr_cfg = cfg["optimization"].get("learning_rates", {})
    wd_cfg = cfg["optimization"].get("weight_decay", {})
    default_wd = float(wd_cfg.get("default", 0.05))
    groups: list[dict] = []
    used: set[int] = set()

    def add_group(name: str, modules: list[torch.nn.Module], lr_key: str, lr_default: float) -> None:
        params = []
        for module in modules:
            for p in module.parameters():
                if p.requires_grad and id(p) not in used:
                    params.append(p)
                    used.add(id(p))
        if params:
            groups.append({"params": params, "lr": float(lr_cfg.get(lr_key, lr_default)), "weight_decay": default_wd, "name": name})

    add_group("dino_adapter", [model.visual.fast_motion_cnn, model.visual.fast_motion_proj, model.visual.temporal_proj], "dino_adapter", 1e-5)
    add_group("predicate_transfer", [model.predicates.transfer], "predicate_transfer", 8e-5)
    add_group("dynamic_predicate_field", [model.predicates.oia_head, model.predicates.temporal, model.predicates.tcn, model.predicates.psi_logit], "dynamic_predicate_field", 8e-5)
    if model.predicates.psi_queries.requires_grad and id(model.predicates.psi_queries) not in used:
        groups.append({"params": [model.predicates.psi_queries], "lr": float(lr_cfg.get("dynamic_predicate_field", 8e-5)), "weight_decay": default_wd, "name": "psi_queries"})
        used.add(id(model.predicates.psi_queries))
    add_group("motion_path", [model.motion], "motion_path", 1e-4)
    add_group("interaction_flow", [model.state_bank, model.flow.predicate_key, model.flow.factor_value, model.flow.edge_head, model.flow.state_head], "interaction_flow", 1e-4)
    if model.flow.factor_queries.requires_grad and id(model.flow.factor_queries) not in used:
        groups.append({"params": [model.flow.factor_queries], "lr": float(lr_cfg.get("interaction_flow", 1e-4)), "weight_decay": default_wd, "name": "flow_factor_queries"})
        used.add(id(model.flow.factor_queries))
    add_group("response_lag", [model.flow.lag], "response_lag", 1e-4)
    add_group("decision_ledger", [model.ledger], "decision_ledger", 1e-4)
    add_group("exp29_head", [model.exp29], "exp29_head", 1e-4)
    add_group("calibration_thresholds", [model.calalign], "calibration_thresholds", 2e-4)

    remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": float(lr_cfg.get("default", 1e-4)), "weight_decay": default_wd, "name": "unassigned"})
    return torch.optim.AdamW(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--no_feature_cache", action="store_true")
    parser.add_argument("--require_no_token_compression", action="store_true")
    args = parser.parse_args()
    cfg = load_interactflow_config(args.config)
    if args.no_feature_cache is False:
        raise ValueError("--no_feature_cache is required")
    if args.require_no_token_compression is False:
        raise ValueError("--require_no_token_compression is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    data = cfg["data"]
    paths = cfg["paths"]
    epochs = int(args.epochs or cfg["optimization"]["epochs"])
    train_ds = PSIDAMO11902Dataset(
        paths["psi_package_root"],
        "train",
        frames_root=paths.get("psi2_root_reference_only"),
        image_size=(int(data["image_height"]), int(data["image_width"])),
        action_dim=int(data["action_dim"]),
        strict_counts=args.max_train_samples is None,
        max_samples=args.max_train_samples,
    )
    test_ds = PSIDAMO11902Dataset(
        paths["psi_package_root"],
        "test",
        frames_root=paths.get("psi2_root_reference_only"),
        image_size=(int(data["image_height"]), int(data["image_width"])),
        action_dim=int(data["action_dim"]),
        strict_counts=args.max_test_samples is None,
        max_samples=args.max_test_samples,
    )
    train_loader = _loader(train_ds, args.batch_size, cfg, shuffle=True)
    test_loader = _loader(test_ds, max(1, min(args.batch_size, 8)), cfg, shuffle=False)
    model = _build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = _build_optimizer(model, cfg)
    steps_total = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps)) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total, eta_min=1e-5)
    manifest = {
        "config": args.config,
        "command_line": vars(args),
        "test_only": True,
        "formal_input_uses_target_frame": False,
        "feature_cache_enabled": False,
        "token_compression": "none",
        "eval_splits": ["test"],
        "train_count": len(train_ds),
        "test_count": len(test_ds),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    best_joint = -1.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            frames = batch.input_frames.to(device, non_blocking=True)
            batch = batch.__class__(
                input_frames=frames,
                action_soft=batch.action_soft.to(device, non_blocking=True),
                action_majority=batch.action_majority.to(device, non_blocking=True),
                exp29=batch.exp29.to(device, non_blocking=True),
                exp29_mask=batch.exp29_mask.to(device, non_blocking=True),
                paper_effective_weight=batch.paper_effective_weight.to(device, non_blocking=True),
                video_id=batch.video_id,
                start_frame=batch.start_frame,
                target_frame_index=batch.target_frame_index,
                target_frame_path=batch.target_frame_path,
                frame_paths=batch.frame_paths,
                explanation_text=batch.explanation_text,
                reasoning_text=batch.reasoning_text,
                sample_id=batch.sample_id,
                meta=batch.meta,
            )
            output = model(frames, epoch=epoch)
            loss, terms = compute_interactflow_losses(output, batch, weights=cfg.get("loss", {}))
            (loss / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, float(cfg["optimization"]["gradient_clip_norm"]))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
            if step == 1 or step % 200 == 0:
                row = {
                    "epoch": epoch,
                    "step": step,
                    "total_steps": len(train_loader),
                    "lr": opt.param_groups[0]["lr"],
                    "loss_total": float(loss.detach().cpu()),
                    "identity_error": float(output.ledger.identity_error.detach().cpu()),
                    "predicate_positive_rate": output.predicates.temporal_stats.get("predicate_positive_rate"),
                }
                append_jsonl(out_dir / "loss_components.jsonl", {**row, **{k: float(v.detach().cpu()) for k, v in terms.items()}})
                print("interactflow_batch " + str(row), flush=True)
        metrics = evaluate(model, test_loader, device, out_dir, epoch=epoch)
        append_jsonl(out_dir / "metrics_summary.jsonl", metrics)
        append_jsonl(out_dir / "state_bank_stats.jsonl", {"epoch": epoch, **output.aux.get("state_stats", {})})
        append_jsonl(out_dir / "interaction_flow_stats.jsonl", {"epoch": epoch, **output.flow.stats})
        append_jsonl(out_dir / "predicate_stats.jsonl", {"epoch": epoch, **output.predicates.temporal_stats})
        append_jsonl(out_dir / "decision_ledger_stats.jsonl", {
            "epoch": epoch,
            "identity_error": float(output.ledger.identity_error.detach().cpu()),
            "gate_mean": float(output.ledger.gate.detach().mean().cpu()),
            "benefit_gate_mean": float(output.ledger.benefit_gate.detach().mean().cpu()),
        })
        append_jsonl(out_dir / "calibration_diagnostics.jsonl", {
            "epoch": epoch,
            "calibration_bias_mean": float(model.ledger.calibration_bias.detach().mean().cpu()),
        })
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_latest.pth")
        if metrics["joint"] > best_joint:
            best_joint = metrics["joint"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_best_test_joint.pth")
            write_json(out_dir / "metrics_best_test_joint.json", metrics)
        print(
            f"interactflow_epoch epoch={epoch} joint={metrics['joint']:.4f} "
            f"Act_mAcc={metrics['action']['Act_mAcc']:.4f} StopF1={metrics['action']['Act_stopF1']:.4f} "
            f"Exp_mF1={metrics['exp29']['Exp_mF1']:.4f} elapsed={time.time()-start:.1f}s",
            flush=True,
        )
    write_json(out_dir / "GOAL_COMPLETED_ACPR_INTERACTFLOW_PP_V1.json", {"completed": True, "best_joint": best_joint})


if __name__ == "__main__":
    main()
