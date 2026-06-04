from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import utils
import vision_transformer as vits
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.bdd100k_scene_state import BDD100KSceneStateIndex, SCENE_STATE_NAMES
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.losses.ceai_losses import ceai_main_loss, ceai_regularizer_losses, compute_total_loss_with_gradient_budget
from fate_oia.losses.pcgrad_lite import apply_pcgrad_lite
from fate_oia.models.ceai_oia_model import CEAIOIAFeatureModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.ceai_artifacts import make_selected_vs_random_evidence_stats, write_json, write_jsonl, json_safe
from fate_oia.utils.ceai_readiness import compute_trainer_readiness_state, default_readiness_state


def load_config_flat(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}
    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
    return flat


def parser_defaults() -> dict[str, Any]:
    return {
        "data_root": "E:/sbw/BDD-OIA",
        "raw_root": "E:/sbw/BDD-OIA",
        "bdd100k_root": "E:/sbw/BDD100K",
        "pretrained_weights": "ckp/reference/dino_deitsmall8_pretrain.pth",
        "checkpoint_key": "teacher",
        "arch": "vit_small",
        "patch_size": 8,
        "n_last_blocks": 1,
        "action_dim": 4,
        "reason_dim": 21,
        "image_height": 360,
        "image_width": 640,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "epochs": 32,
        "num_workers": 4,
        "max_train_samples": 0,
        "max_test_samples": 0,
        "log_every": 200,
        "device": "cuda",
    }


def build_backbone(args, device: torch.device) -> tuple[nn.Module, int]:
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, int(model.embed_dim)


@torch.no_grad()
def extract_tokens(backbone: nn.Module, images: torch.Tensor, n_last_blocks: int) -> torch.Tensor:
    return backbone.get_intermediate_layers(images, n_last_blocks)[-1]


def transform(args):
    return AspectRatioLetterboxTransform(args.image_height, args.image_width, patch_size=args.patch_size, return_meta=True)


def limited(ds, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(ds, list(range(min(max_samples, len(ds)))))
    return ds


def make_loader(args, split: str, shuffle: bool) -> DataLoader:
    ds = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=transform(args),
    )
    ds = limited(ds, args.max_train_samples if split == "train" else args.max_test_samples)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def labels_from_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {"action": batch["action"].float().to(device), "reason": batch["reason"].float().to(device)}


def combined_labels(labels: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([labels["action"], labels["reason"]], dim=1)


def optimizer_for(model: CEAIOIAFeatureModel, args) -> torch.optim.Optimizer:
    groups = [
        {"params": model.base_model.parameters(), "lr": args.lr_base_fate, "name": "base_fate"},
        {"params": model.scene_state.parameters(), "lr": args.lr_scene_state, "name": "scene_state"},
        {"params": list(model.action_expert.parameters()) + list(model.reason_expert.parameters()) + list(model.shared_expert.parameters()), "lr": args.lr_experts, "name": "experts"},
        {"params": list(model.pair_attention.parameters()) + list(model.pair_reliability.parameters()), "lr": args.lr_pair_attention, "name": "pair"},
        {"params": list(model.router.parameters()) + list(model.exchange.parameters()), "lr": args.lr_router, "name": "router"},
        {"params": list(model.tail_expert.parameters()) + list(model.action_set.parameters()), "lr": args.lr_tail, "name": "tail_actionset"},
    ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def set_lrs(optimizer: torch.optim.Optimizer, args, epoch: int) -> float:
    if epoch < args.warmup_epochs:
        scale = (epoch + 1) / max(args.warmup_epochs, 1)
    else:
        t = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        scale = 0.5 * (1.0 + math.cos(math.pi * t))
    bases = [args.lr_base_fate, args.lr_scene_state, args.lr_experts, args.lr_pair_attention, args.lr_router, args.lr_tail]
    for group, base in zip(optimizer.param_groups, bases):
        group["lr"] = max(args.min_lr, float(base) * scale)
    return float(optimizer.param_groups[0]["lr"])


def scene_targets_for(index: BDD100KSceneStateIndex, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    fns = [str(x) for x in batch["file_name"]]
    return index.batch_targets(fns, device=device)


def _mean(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [float(r.get(key, default)) for r in rows if key in r]
    return sum(vals) / max(len(vals), 1)


def _scene_proxy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, float] = {}
    missing = 0
    weak_drivable = 0
    for row in records:
        missing += int(bool(row.get("missing", False)))
        for k, v in (row.get("counts") or {}).items():
            counts[k] = counts.get(k, 0.0) + float(v)
        q = row.get("proxy_quality") or {}
        weak_drivable += int(q.get("direct_drivable_proxy") == "weak_drivable_map_presence")
    n = max(len(records), 1)
    return {
        "available": True,
        "scene_state_names": SCENE_STATE_NAMES,
        "record_count": len(records),
        "missing_count": missing,
        "mean_counts": {k: v / n for k, v in counts.items()},
        "weak_drivable_proxy_count": weak_drivable,
        "geometry_sources": {
            "object_geometry": "box2d_center",
            "lane_geometry": "poly2d_mean_x",
            "direct_drivable_proxy": "weak_drivable_map_presence_if_map_found",
        },
    }


def run_epoch(
    args,
    backbone,
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    train: bool,
    scene_index: BDD100KSceneStateIndex,
    readiness_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.train(train)
    losses = []
    rows = []
    logits_final_a = []
    logits_final_r = []
    logits_base_a = []
    logits_base_r = []
    logits_action_spec = []
    logits_reason_spec = []
    labels_all = []
    file_names: list[str] = []
    diag_acc: dict[str, list[float]] = {}
    scene_records: list[dict[str, Any]] = []
    if train:
        optimizer.zero_grad(set_to_none=True)
    shared_params = model.shared_parameters_for_pcgrad()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = labels_from_batch(batch, device)
        with torch.no_grad():
            tokens = extract_tokens(backbone, images, args.n_last_blocks)
        scene_targets = scene_targets_for(scene_index, batch, device)
        scene_records.extend(list(scene_targets.get("records", [])))
        out = model(tokens, bdd100k_scene_state=scene_targets, readiness_state=readiness_state)
        main = ceai_main_loss(out, labels)
        regs = ceai_regularizer_losses(out, labels, scene_targets, config=vars(args))
        total, stat = compute_total_loss_with_gradient_budget(
            main,
            regs,
            gradient_budget_rho=args.gradient_budget_rho,
            shared_params=shared_params if train else None,
        )
        total_for_backward = total / max(args.gradient_accumulation_steps, 1)
        pc_stats: dict[str, Any] = {
            "pcgrad_task_count": 0,
            "pcgrad_conflict_count": 0,
            "conflict_count": 0,
            "projection_applied_count": 0,
            "pcgrad_mean_dot": 0.0,
            "grad_accumulation_steps": int(args.gradient_accumulation_steps),
            "accumulated_microbatches": 1,
            "overwrote_existing_grad": False,
        }
        if train:
            total_for_backward.backward(retain_graph=bool(args.pcgrad_enabled))
            if args.pcgrad_enabled:
                aux_combined = sum(regs.values()) if regs else total.new_zeros(())
                pc_stats = apply_pcgrad_lite(
                    [main["action_main_loss"], main["reason_main_loss"], aux_combined],
                    shared_params,
                    retain_graph=True,
                    grad_accumulation_steps=args.gradient_accumulation_steps,
                )
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(total.detach().cpu()))
        row = {
            **{k: float(v.detach().cpu()) for k, v in main.items()},
            **stat,
            **pc_stats,
            "epoch": epoch,
            "step": step,
            "split": "train" if train else "test",
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        rows.append(row)
        for k, v in out["diagnostics"].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (float, int, bool)):
                        diag_acc.setdefault(f"{k}.{kk}", []).append(float(vv))
            elif isinstance(v, (float, int, bool)):
                diag_acc.setdefault(k, []).append(float(v))
        logits_final_a.append(out["final_action_logits"].detach().cpu())
        logits_final_r.append(out["final_reason_logits"].detach().cpu())
        logits_base_a.append(out["base_action_logits"].detach().cpu())
        logits_base_r.append(out["base_reason_logits"].detach().cpu())
        logits_action_spec.append(out["action_specialist_logits"].detach().cpu())
        logits_reason_spec.append(out["reason_specialist_logits"].detach().cpu())
        labels_all.append(combined_labels(labels).detach().cpu())
        file_names.extend([str(x) for x in batch["file_name"]])
        if train and args.log_every > 0 and (step + 1) % args.log_every == 0:
            print(json.dumps({"event": "ceai_batch", "epoch": epoch, "batch": step + 1, "loss": losses[-1], "lr": optimizer.param_groups[0]["lr"]}), flush=True)
    if train and rows and (len(rows) % max(args.gradient_accumulation_steps, 1)) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    la = torch.cat(logits_final_a, dim=0) if logits_final_a else torch.empty(0, args.action_dim)
    lr = torch.cat(logits_final_r, dim=0) if logits_final_r else torch.empty(0, args.reason_dim)
    labels = torch.cat(labels_all, dim=0) if labels_all else torch.empty(0, args.action_dim + args.reason_dim)
    metrics = evaluate_snna25(torch.cat([la, lr], dim=1), labels, args.action_dim, threshold_mode="fixed", fixed_threshold=args.fixed_threshold)["metrics"] if labels.numel() else {}
    branch = {}
    if labels.numel():
        branch["base"] = evaluate_snna25(torch.cat([torch.cat(logits_base_a), torch.cat(logits_base_r)], dim=1), labels, args.action_dim, threshold_mode="fixed", fixed_threshold=args.fixed_threshold)["metrics"]
        branch["action_specialist"] = evaluate_snna25(torch.cat([torch.cat(logits_action_spec), torch.cat(logits_base_r)], dim=1), labels, args.action_dim, threshold_mode="fixed", fixed_threshold=args.fixed_threshold)["metrics"]
        branch["reason_specialist"] = evaluate_snna25(torch.cat([torch.cat(logits_base_a), torch.cat(logits_reason_spec)], dim=1), labels, args.action_dim, threshold_mode="fixed", fixed_threshold=args.fixed_threshold)["metrics"]
        branch["final"] = metrics
    diag_mean = {k: sum(v) / max(len(v), 1) for k, v in diag_acc.items()}
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "loss_components": rows,
        "metrics": metrics,
        "branch_metrics": branch,
        "diag": diag_mean,
        "scene_proxy_summary": _scene_proxy_summary(scene_records),
        "logits_action_final": la,
        "logits_reason_final": lr,
        "logits_action_base": torch.cat(logits_base_a) if logits_base_a else torch.empty(0, args.action_dim),
        "logits_reason_base": torch.cat(logits_base_r) if logits_base_r else torch.empty(0, args.reason_dim),
        "labels": labels,
        "file_names": file_names,
    }


def _cfg_get(args, key: str, default: Any = None) -> Any:
    return args.get(key, default) if isinstance(args, dict) else getattr(args, key, default)

def _aggregate_pcgrad(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    return {
        "pcgrad_task_count": int(max([r.get("pcgrad_task_count", 0) for r in rows] or [0])),
        "conflict_count": float(sum(float(r.get("conflict_count", r.get("pcgrad_conflict_count", 0))) for r in rows)),
        "projection_applied_count": float(sum(float(r.get("projection_applied_count", 0)) for r in rows)),
        "grad_accumulation_steps": int(_cfg_get(args, "gradient_accumulation_steps", 1)),
        "accumulated_microbatches": int(len(rows)),
        "overwrote_existing_grad": bool(any(bool(r.get("overwrote_existing_grad", False)) for r in rows)),
    }


def _aggregate_grad_budget(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    return {
        "norm_main": _mean(rows, "norm_main"),
        "norm_aux": _mean(rows, "norm_aux"),
        "budget_scale": _mean(rows, "budget_scale"),
        "rho": float(_cfg_get(args, "gradient_budget_rho", 0.15)),
        "used_true_grad_norm": bool(all(bool(r.get("used_true_grad_norm", False)) for r in rows if "used_true_grad_norm" in r)),
    }


def write_epoch_artifacts(out_dir: Path, epoch: int, row: dict[str, Any], train_stats: dict[str, Any], test_stats: dict[str, Any], manifest: dict[str, Any], readiness_state: dict[str, Any]) -> None:
    e = out_dir / f"epoch_{epoch:03d}"
    e.mkdir(parents=True, exist_ok=True)
    write_json(e / "metrics_summary.json", row)
    write_json(e / "branch_metrics.json", test_stats["branch_metrics"])
    write_jsonl(e / "loss_components.jsonl", train_stats["loss_components"] + test_stats["loss_components"])
    diag = test_stats["diag"]
    files = {
        "readiness_stats.json": {"readiness": readiness_state, "diagnostic_means": {k: v for k, v in diag.items() if "readiness" in k or "q_ar" in k or "pair_attention" in k}},
        "scene_state_stats.json": {k: v for k, v in diag.items() if "scene" in k},
        "scene_state_proxy_stats.json": test_stats["scene_proxy_summary"],
        "implicit_prototype_stats.json": {k: v for k, v in diag.items() if "implicit" in k},
        "action_set_stats.json": {k: v for k, v in diag.items() if "action_set" in k},
        "expert_usage_stats.json": {k: v for k, v in diag.items() if "expert" in k or "token_delta" in k},
        "cross_expert_exchange_stats.json": {k: v for k, v in diag.items() if "cross_expert_exchange" in k or "a2r" in k or "r2a" in k},
        "pair_attention_stats.json": {k: v for k, v in diag.items() if "pair_attention" in k},
        "pair_reliability_stats.json": {k: v for k, v in diag.items() if "pair_reliability" in k or "q_ar" in k or "q_r" in k},
        "router_stats.json": {k: v for k, v in diag.items() if "router" in k or "delta" in k or "readiness_r2a" in k},
        "grad_conflict_stats.json": _aggregate_pcgrad(train_stats["loss_components"], manifest["config_resolved"]),
        "pcgrad_accum_stats.json": _aggregate_pcgrad(train_stats["loss_components"], manifest["config_resolved"]),
        "grad_budget_stats.json": _aggregate_grad_budget(train_stats["loss_components"], manifest["config_resolved"]),
        "bdd100k_scene_state_stats.json": test_stats["scene_proxy_summary"],
        "selected_vs_random_evidence_stats.json": make_selected_vs_random_evidence_stats(None, None, computed=False),
        "run_manifest_epoch.json": manifest,
    }
    for name, obj in files.items():
        write_json(e / name, obj if obj else {"available": True})
    write_jsonl(e / "failure_cases.jsonl", [])


def build_parser() -> argparse.ArgumentParser:
    d = parser_defaults()
    ap = argparse.ArgumentParser(description="Train CEAI-OIA V1.1 with direct image DINO tokens and test-only evaluation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    for k, v in d.items():
        t = type(v) if v is not None else str
        ap.add_argument(f"--{k}", type=t, default=v)
    ap.add_argument("--feature_cache_enabled", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--test_only_evaluation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--best_selection_split", default="test")
    ap.add_argument("--fixed_threshold", type=float, default=0.5)
    ap.add_argument("--lr_base_fate", type=float, default=2e-4)
    ap.add_argument("--lr_scene_state", type=float, default=2.5e-4)
    ap.add_argument("--lr_experts", type=float, default=3e-4)
    ap.add_argument("--lr_pair_attention", type=float, default=2.5e-4)
    ap.add_argument("--lr_router", type=float, default=2e-4)
    ap.add_argument("--lr_tail", type=float, default=2.5e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gradient_budget_rho", type=float, default=0.15)
    ap.add_argument("--pcgrad_enabled", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    cfg = load_config_flat(args.config)
    cli = {tok[2:] for tok in sys.argv[1:] if tok.startswith("--")}
    for k, v in cfg.items():
        if hasattr(args, k) and k not in cli:
            setattr(args, k, v)
    if args.feature_cache_enabled:
        raise RuntimeError("CEAI run requires feature_cache_enabled=false")
    if not args.test_only_evaluation or args.best_selection_split != "test":
        raise RuntimeError("CEAI run requires test-only evaluation and best-on-test")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "args.json", vars(args))
    backbone, dim = build_backbone(args, device)
    model = CEAIOIAFeatureModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim).to(device)
    opt = optimizer_for(model, args)
    train_loader = make_loader(args, "train", True)
    test_loader = make_loader(args, "test", False)
    scene_index = BDD100KSceneStateIndex(args.bdd100k_root)
    manifest = {
        "repo_name": "FATE-OIA",
        "method": "CEAI-OIA V1.1",
        "hostname": socket.gethostname(),
        "command": " ".join(sys.argv),
        "feature_cache_enabled": False,
        "token_compression": "none",
        "eval_splits": ["test"],
        "best_selection_split": "test",
        "pretrained_weights": args.pretrained_weights,
        "train_split_count": len(train_loader.dataset),
        "test_split_count": len(test_loader.dataset),
        "config_resolved": vars(args),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    best = -1.0
    history = []
    readiness_state = default_readiness_state()
    action_drop_epochs = 0
    for epoch in range(args.epochs):
        lr_now = set_lrs(opt, args, epoch)
        train_stats = run_epoch(args, backbone, model, train_loader, opt, device, epoch, True, scene_index, readiness_state=readiness_state)
        with torch.no_grad():
            test_stats = run_epoch(args, backbone, model, test_loader, opt, device, epoch, False, scene_index, readiness_state=readiness_state)
        tm = test_stats["metrics"]
        joint = 0.5 * float(tm.get("Act_mF1", 0.0)) + 0.5 * float(tm.get("Exp_mF1", 0.0))
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "test_loss": test_stats["loss"], "joint_test_score": joint, "test_metrics": tm, "branch_metrics": test_stats["branch_metrics"], "diag": test_stats["diag"], "readiness_state_used": readiness_state, "lr": lr_now}
        history.append(row)
        with (out_dir / "metrics_summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        write_json(out_dir / "metrics_summary.json", row)
        write_json(out_dir / "metrics_latest.json", row)
        write_epoch_artifacts(out_dir, epoch, row, train_stats, test_stats, manifest, readiness_state)
        latest = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args), "dim": dim, "best_test_score": max(best, joint)}
        torch.save(latest, out_dir / "checkpoint_latest.pth")
        torch.save(test_stats["logits_action_final"], out_dir / "logits_action_final_test.pt")
        torch.save(test_stats["logits_reason_final"], out_dir / "logits_reason_final_test.pt")
        torch.save(test_stats["logits_action_base"], out_dir / "logits_action_base_test.pt")
        torch.save(test_stats["logits_reason_base"], out_dir / "logits_reason_base_test.pt")
        torch.save(test_stats["labels"], out_dir / "labels_test.pt")
        write_json(out_dir / "file_names_test.json", test_stats["file_names"])
        if joint >= best:
            best = joint
            torch.save(latest, out_dir / "checkpoint_best_test.pth")
            torch.save(latest, out_dir / "checkpoint_best.pth")
            write_json(out_dir / "metrics_best_test.json", row)
        next_state = compute_trainer_readiness_state(row, previous_action_drop_epochs=action_drop_epochs, evidence_gate_ok=False, evidence_not_used_for_action=True)
        action_drop_epochs = int(next_state.get("action_drop_epochs", 0))
        readiness_state = next_state
        print(json.dumps({"event": "ceai_epoch", "epoch": epoch, "joint": joint, "Act_mF1": tm.get("Act_mF1"), "Exp_mF1": tm.get("Exp_mF1"), "Exp_mAP": tm.get("Exp_mAP"), "lr": lr_now, "next_r2a_active": readiness_state.get("r2a_active"), "router_action_scale": readiness_state.get("router_action_scale")}), flush=True)
    write_json(out_dir / "history.json", history)


if __name__ == "__main__":
    main()
