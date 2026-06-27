from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fate_oia.acpr_interactflow.artifacts import append_jsonl, write_json
from fate_oia.acpr_interactflow.config import load_interactflow_config
from fate_oia.acpr_interactflow.interventions import evaluate_intervention_suite
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
    pred_cfg = cfg["model"].get("predicates", {})
    return ACPRInteractFlowPPModel(
        pretrained_weights=cfg["paths"]["dino_weights"],
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path=cfg["model"]["interaction_flow"]["grammar_yaml"],
        exp29_names_path=cfg["paths"].get("psi_label_embedding_json"),
        oia_acpr_checkpoint=cfg["paths"].get("oia_acpr_checkpoint"),
        text_encoder_model=cfg["paths"].get("text_encoder_model"),
        require_oia_transfer_source=bool(pred_cfg.get("require_oia_transfer_source", False)),
        require_transformer_text=bool(pred_cfg.get("require_transformer_text", False)),
        action_dim=int(cfg["data"]["action_dim"]),
        dino_chunk_size=int(cfg["model"]["visual_encoder"].get("dino_chunk_size", 2)),
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


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_epoch_prediction_artifacts(run_dir: Path, epoch_dir: Path) -> None:
    import json

    logits_action_path = run_dir / "logits_action_test.pt"
    logits_exp_path = run_dir / "logits_exp29_test.pt"
    labels_action_path = run_dir / "labels_action_test.pt"
    labels_exp_path = run_dir / "labels_exp29_test.pt"
    files_path = run_dir / "file_names_test.json"
    if not all(p.exists() for p in [logits_action_path, logits_exp_path, labels_action_path, labels_exp_path, files_path]):
        return
    action_logits = torch.load(logits_action_path, map_location="cpu")
    exp_logits = torch.load(logits_exp_path, map_location="cpu")
    action_labels = torch.load(labels_action_path, map_location="cpu")
    exp_labels = torch.load(labels_exp_path, map_location="cpu")
    files = json.loads(files_path.read_text(encoding="utf-8"))
    action_probs = torch.softmax(action_logits, dim=-1)
    exp_probs = torch.sigmoid(exp_logits)
    action_rows = []
    exp_rows = []
    for i, file_name in enumerate(files):
        action_rows.append(
            {
                "epoch": epoch_dir.name,
                "sample_id": file_name,
                "pred_action": int(action_probs[i].argmax()),
                "gt_action": int(action_labels[i]),
                "action_probs": action_probs[i].tolist(),
            }
        )
        top_exp = torch.topk(exp_probs[i], k=min(5, exp_probs.shape[1])).indices.tolist()
        exp_rows.append(
            {
                "epoch": epoch_dir.name,
                "sample_id": file_name,
                "top_exp29": [int(x) for x in top_exp],
                "exp29_probs_top5": [float(exp_probs[i, x]) for x in top_exp],
                "positive_exp29_indices": [int(x) for x in torch.nonzero(exp_labels[i] > 0.5, as_tuple=False).flatten().tolist()],
            }
        )
    _write_jsonl_rows(epoch_dir / "predictions_action.jsonl", action_rows)
    _write_jsonl_rows(epoch_dir / "predictions_exp29.jsonl", exp_rows)


def _write_epoch_artifacts(
    run_dir: Path,
    epoch: int,
    metrics: dict,
    output,
    model: ACPRInteractFlowPPModel,
    loss_rows: list[dict],
    grad_rows: list[dict],
    influence: dict,
) -> None:
    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    write_json(epoch_dir / "action_metrics.json", metrics["action"])
    write_json(epoch_dir / "exp29_metrics.json", metrics["exp29"])
    write_json(epoch_dir / "joint_metrics.json", {"epoch": epoch, "joint": metrics["joint"], "formula": "0.60*Act_mAcc + 0.25*Stop_F1 + 0.15*Exp_mF1"})
    _write_jsonl_rows(epoch_dir / "loss_components.jsonl", loss_rows)
    write_json(epoch_dir / "gradient_norms.json", {"epoch": epoch, "optimizer_steps": grad_rows})
    write_json(epoch_dir / "predicate_stats.json", {"epoch": epoch, **output.predicates.temporal_stats})
    write_json(epoch_dir / "nnpu_calibration.json", {
        "epoch": epoch,
        "action_temperature": torch.exp(model.calalign.action_temp.clamp(-1.0, 1.0)).detach().cpu().tolist(),
        "action_bias": model.calalign.action_bias.clamp(-2, 2).detach().cpu().tolist(),
        "exp29_temperature": torch.exp(model.calalign.exp_temp.clamp(-1.0, 1.0)).detach().cpu().tolist(),
        "exp29_bias": model.calalign.exp_bias.clamp(-2, 2).detach().cpu().tolist(),
    })
    write_json(epoch_dir / "interaction_state_stats.json", {"epoch": epoch, **output.aux.get("state_stats", {})})
    write_json(epoch_dir / "response_lag_stats.json", {"epoch": epoch, **output.flow.stats})
    write_json(epoch_dir / "decision_ledger_stats.json", {
        "epoch": epoch,
        "identity_error": float(output.ledger.identity_error.detach().cpu()),
        "gate_mean": float(output.ledger.gate.detach().mean().cpu()),
        "benefit_gate_mean": float(output.ledger.benefit_gate.detach().mean().cpu()),
    })
    write_json(epoch_dir / "lightweight_interaction_influence.json", influence)
    _write_epoch_prediction_artifacts(run_dir, epoch_dir)
    fixed_rows = []
    for i in range(min(8, output.action_logits.shape[0])):
        fixed_rows.append(
            {
                "epoch": epoch,
                "index": i,
                "action_logits": output.action_logits[i].detach().cpu().tolist(),
                "global_logits": output.ledger.global_logits[i].detach().cpu().tolist(),
                "flow_delta_logits": output.ledger.flow_delta_logits[i].detach().cpu().tolist(),
                "calibration_delta": output.ledger.calibration_delta[i].detach().cpu().tolist(),
                "identity_error": float(output.ledger.identity_error.detach().cpu()),
            }
        )
    _write_jsonl_rows(epoch_dir / "fixed_case_intermediate_outputs.jsonl", fixed_rows)


def _build_warmup_cosine_scheduler(opt: torch.optim.Optimizer, cfg: dict, total_updates: int):
    opt_cfg = cfg["optimization"]
    warmup_updates = max(1, int(float(opt_cfg.get("warmup_ratio", 0.05)) * total_updates))
    min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.10))

    def lr_lambda(step: int) -> float:
        if step < warmup_updates:
            return max((step + 1) / warmup_updates, min_lr_ratio)
        progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda), warmup_updates


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps)))
    steps_total = updates_per_epoch * epochs
    sched, warmup_updates = _build_warmup_cosine_scheduler(opt, cfg, steps_total)
    precision = str(cfg["optimization"].get("precision", "fp32")).lower()
    use_bf16 = precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
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
        "precision": precision,
        "bf16_autocast_enabled": bool(use_bf16),
        "scheduler": cfg["optimization"].get("scheduler", "cosine"),
        "warmup_ratio": cfg["optimization"].get("warmup_ratio", 0.05),
        "warmup_updates": warmup_updates,
        "optimizer_groups": [{"name": g.get("name"), "lr": g.get("lr"), "weight_decay": g.get("weight_decay")} for g in opt.param_groups],
        "oia_transfer_source_loaded": bool(model.predicates.transfer.report().get("source_loaded", False)),
        "oia_transfer_text_embedding_source": model.predicates.transfer.report().get("text_embedding_source"),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_json(out_dir / "oia_transfer_report.json", model.predicates.transfer.report())
    write_json(out_dir / "optimizer_groups.json", manifest["optimizer_groups"])
    write_json(out_dir / "git_provenance.json", {"git_head": _git_head(), "config": args.config})
    config_path = Path(args.config)
    if config_path.exists():
        (out_dir / "config_resolved.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    best_joint = -1.0
    best_action = -1.0
    best_exp = -1.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        start = time.time()
        epoch_loss_rows: list[dict] = []
        epoch_grad_rows: list[dict] = []
        last_frames_for_audit: torch.Tensor | None = None
        for step, batch in enumerate(train_loader, start=1):
            frames = batch.input_frames.to(device, non_blocking=True)
            last_frames_for_audit = frames[:1].detach()
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
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                output = model(frames, epoch=epoch)
                loss, terms = compute_interactflow_losses(output, batch, weights=cfg.get("loss", {}))
            (loss / args.gradient_accumulation_steps).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["optimization"]["gradient_clip_norm"]))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                epoch_grad_rows.append(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_step": step,
                        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                        "lr": opt.param_groups[0]["lr"],
                    }
                )
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
                loss_weights = cfg.get("loss", {})
                term_values = {k: float(v.detach().cpu()) for k, v in terms.items()}
                weighted_values = {
                    f"{k}_weighted": float(loss_weights.get(k, 1.0 if k == "total_loss" else 0.0)) * val
                    for k, val in term_values.items()
                    if k != "total_loss"
                }
                log_row = {**row, **term_values, **weighted_values}
                append_jsonl(out_dir / "loss_components.jsonl", log_row)
                epoch_loss_rows.append(log_row)
                print("interactflow_batch " + str(row), flush=True)
        metrics = evaluate(model, test_loader, device, out_dir, epoch=epoch)
        influence = {"epoch": epoch, "available": False}
        if last_frames_for_audit is not None:
            try:
                with torch.no_grad():
                    influence = evaluate_intervention_suite(model, last_frames_for_audit.to(device), epoch=epoch)
                    influence["available"] = True
            except Exception as exc:
                influence = {"epoch": epoch, "available": False, "error": repr(exc)}
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
        if device.type == "cuda":
            append_jsonl(out_dir / "gpu_memory.jsonl", {
                "epoch": epoch,
                "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
                "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
            })
        _write_epoch_artifacts(out_dir, epoch, metrics, output, model, epoch_loss_rows, epoch_grad_rows, influence)
        ckpt = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}
        _atomic_torch_save(ckpt, out_dir / "checkpoint_latest.pth")
        if metrics["joint"] > best_joint:
            best_joint = metrics["joint"]
            _atomic_torch_save(ckpt, out_dir / "checkpoint_best_joint.pth")
            _atomic_torch_save(ckpt, out_dir / "checkpoint_best_test.pth")
            _atomic_torch_save(ckpt, out_dir / "checkpoint_best_test_joint.pth")
            write_json(out_dir / "metrics_best_test_joint.json", metrics)
        action_score = float(metrics["action"]["Act_mAcc"])
        exp_score = float(metrics["exp29"]["Exp_mF1"])
        if action_score > best_action:
            best_action = action_score
            _atomic_torch_save(ckpt, out_dir / "checkpoint_best_action.pth")
        if exp_score > best_exp:
            best_exp = exp_score
            _atomic_torch_save(ckpt, out_dir / "checkpoint_best_exp.pth")
        print(
            f"interactflow_epoch epoch={epoch} joint={metrics['joint']:.4f} "
            f"Act_mAcc={metrics['action']['Act_mAcc']:.4f} StopF1={metrics['action']['Act_stopF1']:.4f} "
            f"Exp_mF1={metrics['exp29']['Exp_mF1']:.4f} elapsed={time.time()-start:.1f}s",
            flush=True,
        )
    completion = {"completed": True, "best_joint": best_joint, "best_action": best_action, "best_exp": best_exp}
    write_json(out_dir / "run_complete.json", completion)
    write_json(out_dir / "GOAL_COMPLETED_ACPR_INTERACTFLOW_PP_V1.json", completion)


if __name__ == "__main__":
    main()
